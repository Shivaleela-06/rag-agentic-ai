import os
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    validate_config,
)

GDRIVE_FILE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"
GDRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"


def ensure_pdf_downloaded(target_path: str = PDF_PATH) -> str:
    """
    Ensure the Agentic AI eBook PDF is downloaded and present on disk.
    If not, downloads it from the provided Google Drive source link.
    """
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        print(f"[Ingestion] PDF not found at {path}. Downloading from Google Drive...")
        urllib.request.urlretrieve(GDRIVE_DOWNLOAD_URL, str(path))
        print(f"[Ingestion] Download complete. File size: {path.stat().st_size} bytes.")
    else:
        print(f"[Ingestion] Found existing PDF at {path} ({path.stat().st_size} bytes).")

    return str(path)


def setup_pinecone_index(
    index_name: str = PINECONE_INDEX_NAME,
    api_key: str = PINECONE_API_KEY,
    dimension: int = EMBEDDING_DIMENSION,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
):
    """
    Check if Pinecone index exists. If not, create a new serverless index.
    """
    if not api_key:
        raise ValueError("PINECONE_API_KEY is not configured.")

    pc = Pinecone(api_key=api_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"[Ingestion] Creating Pinecone index '{index_name}' (dim={dimension}, metric={metric})...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"[Ingestion] Index '{index_name}' successfully created. Waiting for it to become ready...")
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(2)
        print(f"[Ingestion] Index '{index_name}' is ready.")
    else:
        print(f"[Ingestion] Pinecone index '{index_name}' already exists.")

    return pc


def load_and_split_pdf(
    pdf_path: str = PDF_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """
    Load pages from the PDF document and split into text chunks.
    """
    ensure_pdf_downloaded(pdf_path)

    print(f"[Ingestion] Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"[Ingestion] Successfully loaded {len(docs)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[Ingestion] Split into {len(chunks)} chunks (chunk_size={chunk_size}, chunk_overlap={chunk_overlap}).")
    return chunks


def run_ingestion(
    pdf_path: str = PDF_PATH,
    index_name: str = PINECONE_INDEX_NAME,
    batch_size: int = 100,
):
    """
    Full document ingestion and vector storage pipeline:
    1. Verify PDF document
    2. Split PDF into chunks
    3. Ensure Pinecone index exists
    4. Upsert embeddings into Pinecone
    """
    validate_config(require_keys=True)

    # 1. Setup Index
    setup_pinecone_index(index_name=index_name)

    # 2. Split Document
    chunks = load_and_split_pdf(pdf_path=pdf_path)

    # 3. Create Embeddings & Store in Pinecone
    print(f"[Ingestion] Initializing OpenAI embeddings with model '{EMBEDDING_MODEL}'...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    print(f"[Ingestion] Upserting {len(chunks)} chunks into Pinecone index '{index_name}' in batches of {batch_size}...")
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )

    # Batch upsert to handle rate limits cleanly
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_store.add_documents(documents=batch)
        print(f"[Ingestion] Processed chunks {i + 1} to {min(i + batch_size, len(chunks))} / {len(chunks)}...")

    print(f"[Ingestion] Ingestion pipeline successfully completed! {len(chunks)} chunks indexed in '{index_name}'.")
    return vector_store


if __name__ == "__main__":
    import sys
    print("--- Starting Document Ingestion Pipeline ---")
    try:
        run_ingestion()
    except Exception as e:
        print(f"[Error] Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)
