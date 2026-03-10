"""
Improvements over baseline:
  1. History-aware query rewriting  — collapses conversation context into the query
  2. Query expansion               — generates 3 sub-queries -> broader recall
  3. Larger retrieval k=8 per query -> more candidates before re-ranking
  4. LLM re-ranking                -> selects the best 5 chunks from candidates
  5. Richer system prompt          -> date-aware, completeness-focused
  6. Matching encoder to ingest    -> BAAI/bge-base-en-v1.5
"""

from datetime import date
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv(override=True)

# Config 
MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

RETRIEVAL_K = 8          # candidates per sub-query
EXPANSION_QUERIES = 3    # number of sub-queries to generate
RERANK_TOP_N = 5         # chunks to keep after re-ranking

# Models 
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
llm = ChatOpenAI(temperature=0, model_name=MODEL)

# Prompts 
REWRITE_PROMPT = """You are helping a RAG system retrieve relevant documents.
Given a conversation history and the latest user question, rewrite the question
into a single self-contained query that includes all necessary context from history.
Return ONLY the rewritten query, nothing else."""

EXPANSION_PROMPT = """You are helping a RAG system retrieve relevant documents from an
insurance company knowledge base (InsureLLM).
Given a user query, generate {n} distinct search queries that together maximize the
chance of retrieving all relevant information. Cover different phrasings and aspects.
Return ONLY the queries, one per line, no numbering or preamble."""

RERANK_PROMPT = """You are a relevance judge for a RAG system.
Below is a user question followed by {n} retrieved text chunks (numbered).
Select the {top} most relevant chunk numbers that best answer the question.
Return ONLY a comma-separated list of numbers (e.g. 1,3,5). No explanation."""

SYSTEM_PROMPT = """You are a knowledgeable, thorough assistant representing InsureLLM.
Today's date is {today}.

Use the retrieved context below to answer the user's question as completely as possible.
Guidelines:
- Be specific: include names, dates, figures, and product details from the context.
- Be complete: if multiple items are relevant, list ALL of them, not just a few.
- If the context contains partial information, say what you know and flag what's missing.
- If you genuinely don't know, say so clearly — do not hallucinate.
- Structure longer answers with bullet points or short paragraphs for readability.

Retrieved Context:
{context}
"""


# History-aware query rewriting
def rewrite_with_history(question: str, history: list[dict]) -> str:
    """
    If there's conversation history, rewrite the question so it's fully
    self-contained and resolves things like pronouns, references, etc.
    """
    if not history:
        return question
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:] 
    )
    response = llm.invoke([
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=f"Conversation history:\n{history_text}\n\nLatest question: {question}"),
    ])
    rewritten = response.content.strip()
    return rewritten if rewritten else question


# Step 2: Query expansion 
def expand_query(question: str) -> list[str]:
    """
    Generate multiple sub-queries from the original question to maximise recall.
    Always includes the original question as the first query.
    """
    prompt = EXPANSION_PROMPT.format(n=EXPANSION_QUERIES)
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=question),
    ])
    lines = [ln.strip() for ln in response.content.strip().splitlines() if ln.strip()]
    # Deduplicate while keeping order; always keep original
    seen = {question}
    sub_queries = [question]
    for line in lines:
        if line not in seen:
            seen.add(line)
            sub_queries.append(line)
    return sub_queries[:EXPANSION_QUERIES + 1]


# Multi-query retrieval with dedup 
def multi_query_retrieve(queries: list[str]) -> list[Document]:
    """
    Retrieve RETRIEVAL_K docs for each sub-query, deduplicate by page_content.
    For summary chunks, substitute the original_text so the LLM gets full context.
    """
    seen_content = set()
    all_docs = []
    for query in queries:
        docs = retriever.invoke(query)
        for doc in docs:
            # If this is a summary chunk, swap in the richer original text
            content = doc.metadata.get("original_text") or doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                # Return a copy with the full text
                full_doc = Document(
                    page_content=content,
                    metadata=doc.metadata,
                )
                all_docs.append(full_doc)
    return all_docs


# LLM re-ranking 
def rerank(question: str, docs: list[Document]) -> list[Document]:
    """
    Ask the LLM to pick the RERANK_TOP_N most relevant chunks from candidates.
    Falls back gracefully if re-ranking fails.
    """
    if len(docs) <= RERANK_TOP_N:
        return docs

    numbered = "\n\n".join(
        f"[{i+1}] {doc.page_content[:600]}" for i, doc in enumerate(docs)
    )
    prompt = RERANK_PROMPT.format(n=len(docs), top=RERANK_TOP_N)
    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Question: {question}\n\nChunks:\n{numbered}"),
        ])
        indices_str = response.content.strip()
        indices = [int(x.strip()) - 1 for x in indices_str.split(",") if x.strip().isdigit()]
        # Validate and deduplicate
        valid = [i for i in indices if 0 <= i < len(docs)]
        # If we got enough valid indices, use them; else fall back to top-N
        if len(valid) >= 3:
            selected = [docs[i] for i in valid[:RERANK_TOP_N]]
            return selected
    except Exception as e:
        print(f"  Re-ranking failed ({e}), using top-{RERANK_TOP_N} by retrieval order")

    return docs[:RERANK_TOP_N]


# Public API 
def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question (no history, no LLM calls).
    Used by the evaluator's retrieval scoring.
    """
    sub_queries = expand_query(question)
    candidates = multi_query_retrieve(sub_queries)
    return rerank(question, candidates)


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    CompleteRAG pipeline:
    rewrite -> expand -> retrieve -> rerank -> answer
    Returns (answer_text, context_docs).
    """
    # Resolve history references into a self-contained query
    effective_query = rewrite_with_history(question, history)

    # Expand into sub-queries and retrieve
    sub_queries = expand_query(effective_query)
    candidates = multi_query_retrieve(sub_queries)

    # Re-rank to select best chunks
    top_docs = rerank(effective_query, candidates)

    # Build context string with source labels
    context_parts = []
    for doc in top_docs:
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("doc_type", "")
        label = f"[{doc_type} | {source}]"
        context_parts.append(f"{label}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # Build prompt and call LLM
    system_prompt = SYSTEM_PROMPT.format(
        today=date.today().strftime("%B %d, %Y"),
        context=context,
    )
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content, top_docs
