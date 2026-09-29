import os
from pathlib import Path
from dotenv import load_dotenv

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# API Keys & Services
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

# Models and Vector Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = 1536  # Dimension for text-embedding-3-small
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Chunking & Retrieval Parameters
PDF_PATH = os.getenv("PDF_PATH", str(PROJECT_ROOT / "data" / "Ebook-Agentic-AI.pdf"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "3"))

# Server Parameters
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


def validate_config(require_keys: bool = False):
    """
    Validate that required environment variables are set.
    """
    missing = []
    if require_keys:
        if not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not PINECONE_API_KEY:
            missing.append("PINECONE_API_KEY")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please configure them in your .env file or environment."
            )
    return True
