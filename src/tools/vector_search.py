from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from langchain_chroma import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

ROOT = Path(__file__).resolve().parents[3]
DB_DIR = ROOT / "data" / "chroma"
COLLECTION = "paperbuddy"
EMBED = "sentence-transformers/all-MiniLM-L6-v2"

def _db(db_dir: Path | str = DB_DIR) -> Chroma:
    emb = HuggingFaceEmbeddings(model_name=EMBED)
    return Chroma(collection_name=COLLECTION, embedding_function=emb, persist_directory=str(db_dir))

def search(query: str, k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
    return _db().similarity_search_with_relevance_scores(query, k=k, filter=where)

def search_mmr(query: str, k: int = 6, fetch_k: int = 30, where: Optional[Dict[str, Any]] = None) -> List[Document]:
    return _db().max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, filter=where)
