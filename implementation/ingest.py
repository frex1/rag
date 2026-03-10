"""
Enhanced RAG ingestion pipeline for InsureLLM
Improvements over baseline:
  1. Better encoder: BAAI/bge-base-en-v1.5 -> (768-dim, stronger retrieval)
  2. Smarter chunking: smaller chunks (600/100) + metadata injected into text
  3. Dual indexing: raw chunks + LLM-generated summaries stored together
"""

import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv(override=True)

# Config 
MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# Upgraded encoder with 768-dim -> significantly stronger than all-MiniLM-L6-v2
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},   # required for BGE cosine similarity
)

llm = ChatOpenAI(temperature=0, model_name=MODEL)

SUMMARY_SYSTEM_PROMPT = """You are a precise document summarizer for an insurance company knowledge base.
Given a text chunk, write a concise factual summary (2-4 sentences) that:
- Captures the key facts, names, numbers, and relationships
- Uses plain language
- Retains all important entities (people, products, dates, amounts)
Do NOT add any information not present in the chunk.
Respond with ONLY the summary, no preamble."""


# Load raw documents 
def fetch_documents() -> list[Document]:
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            # Inject filename into metadata for later provenance
            doc.metadata["filename"] = os.path.basename(doc.metadata.get("source", ""))
            documents.append(doc)
    print(f"Loaded {len(documents)} source documents")
    return documents


# Chunk with metadata prefix
def create_chunks(documents: list[Document]) -> list[Document]:
    """
    Smaller chunks (600 chars / 100 overlap) to avoid splitting related facts.
    Each chunk's page_content is prefixed with doc metadata so the encoder
    sees the document type and filename as part of the semantic signal.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    raw_chunks = splitter.split_documents(documents)

    enriched = []
    for chunk in raw_chunks:
        doc_type = chunk.metadata.get("doc_type", "unknown")
        filename = chunk.metadata.get("filename", "")
        # Prefix gives the encoder richer semantic context
        prefix = f"[Source: {doc_type} | File: {filename}]\n"
        chunk.page_content = prefix + chunk.page_content
        chunk.metadata["chunk_type"] = "raw"
        enriched.append(chunk)

    print(f"Created {len(enriched)} raw chunks")
    return enriched


# Generate LLM summaries for dual indexing
def generate_summary(text: str) -> str:
    """Ask the LLM to produce a concise factual summary of a chunk."""
    try:
        response = llm.invoke([
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=text),
        ])
        return response.content.strip()
    except Exception as e:
        print(f"  Warning: summary generation failed ({e}), using original text")
        return text


def create_summary_chunks(raw_chunks: list[Document]) -> list[Document]:
    """
    For each raw chunk, generate an LLM summary and store it as a separate
    Document that points back to the same source metadata.
    This helps holistic / spanning / relationship queries that need
    high-level semantic matching rather than keyword matching.
    """
    summary_chunks = []
    print(f"Generating summaries for {len(raw_chunks)} chunks (this may take a while)...")
    for i, chunk in enumerate(raw_chunks):
        if i % 20 == 0:
            print(f"  Summarising chunk {i+1}/{len(raw_chunks)}...")
        summary_text = generate_summary(chunk.page_content)
        summary_doc = Document(
            page_content=summary_text,
            metadata={
                **chunk.metadata,
                "chunk_type": "summary",
                "original_text": chunk.page_content,  # store original for retrieval
            },
        )
        summary_chunks.append(summary_doc)
    print(f"Created {len(summary_chunks)} summary chunks")
    return summary_chunks


# Build vector store
def create_embeddings(all_chunks: list[Document]) -> Chroma:
    """Embed all chunks (raw + summary) into a single Chroma collection."""
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )

    collection = vectorstore._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"Vectorstore: {count:,} vectors × {dimensions:,} dimensions")
    return vectorstore


# Main 
if __name__ == "__main__":
    print("*** InsureLLM RAG Ingestion Pipeline ***\n")

    documents = fetch_documents()
    raw_chunks = create_chunks(documents)
    summary_chunks = create_summary_chunks(raw_chunks)

    all_chunks = raw_chunks + summary_chunks
    print(f"\nTotal chunks to index: {len(all_chunks)}")

    create_embeddings(all_chunks)
    print("\n*** Ingestion complete ***")
