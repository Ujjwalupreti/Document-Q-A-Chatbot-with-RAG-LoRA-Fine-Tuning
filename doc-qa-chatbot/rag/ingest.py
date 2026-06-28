"""
Document ingestion pipeline: load PDF/DOCX/TXT files, split into chunks,
embed with sentence-transformers, and persist to a per-session Chroma store.
"""
import logging
import shutil
import threading
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
CHROMA_ROOT = Path("./chroma_db")

# R-01: Thread-safe singleton initialization — prevents race conditions
# when multiple uploads run concurrently via run_in_threadpool.
_lock = threading.Lock()
_embeddings = None
_splitter = None


def _get_loader(file_path: str):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(file_path)
    if suffix == ".docx":
        return Docx2txtLoader(file_path)
    if suffix == ".txt":
        return TextLoader(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        with _lock:
            # Double-check after acquiring the lock.
            if _embeddings is None:
                _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_splitter() -> RecursiveCharacterTextSplitter:
    """
    Token-aware splitter: chunk_size/chunk_overlap are counted in *tokens*
    (via tiktoken's cl100k_base encoding) rather than raw characters, so a
    "512-token chunk with 64-token overlap" is literally true.

    Note: the first call downloads the cl100k_base encoding (~1.7MB) and
    caches it locally — requires internet access on first run.
    """
    global _splitter
    if _splitter is None:
        with _lock:
            if _splitter is None:
                _splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    encoding_name="cl100k_base",
                    chunk_size=CHUNK_SIZE_TOKENS,
                    chunk_overlap=CHUNK_OVERLAP_TOKENS,
                )
    return _splitter


def ingest_document(file_path: str, session_id: str) -> int:
    """
    Load a document, chunk it (512 tokens, 64-token overlap), embed the
    chunks, and store them in a per-session persistent Chroma collection.

    Returns the number of chunks stored.
    """
    # R-05: Wrap loader in try/except to surface useful error messages
    # for corrupt, password-protected, or unreadable files.
    try:
        loader = _get_loader(file_path)
        documents = loader.load()
    except Exception as exc:
        raise ValueError(f"Failed to load document: {exc}") from exc

    if not documents:
        raise ValueError("Document loaded but contained no text content.")

    chunks = get_splitter().split_documents(documents)

    persist_dir = CHROMA_ROOT / session_id
    persist_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=session_id,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )
    vectorstore.add_documents(chunks)

    return len(chunks)


def get_vectorstore(session_id: str) -> Chroma:
    persist_dir = CHROMA_ROOT / session_id
    return Chroma(
        collection_name=session_id,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )


def delete_session(session_id: str) -> None:
    """R-04: Delete both the Chroma collection and the on-disk persist dir."""
    persist_dir = CHROMA_ROOT / session_id
    # Delete the Chroma collection first to clear any in-memory cache.
    try:
        store = Chroma(
            collection_name=session_id,
            embedding_function=get_embeddings(),
            persist_directory=str(persist_dir),
        )
        store.delete_collection()
    except Exception as exc:
        logger.warning("Could not delete Chroma collection for session %s: %s", session_id, exc)

    # Then remove the on-disk files.
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
