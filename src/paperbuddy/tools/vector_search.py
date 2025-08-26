from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import math

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
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=emb,
        persist_directory=str(db_dir),
    )

def _to_similarity(score: float) -> float:
    # Se vier distância de cosseno (0..2), mapeia p/ similaridade (1 - d/2)
    if 0.0 <= score <= 2.0:
        return max(0.0, min(1.0, 1.0 - (score / 2.0)))
    # Se vier inner product/valores negativos, usa sigmoid como similaridade 0..1
    return 1.0 / (1.0 + math.exp(-score))

def search(query: str, k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
    # Usa with_score (distância) e converte para similaridade 0..1
    pairs = _db().similarity_search_with_score(query, k=k, filter=where)
    return [(doc, _to_similarity(score)) for doc, score in pairs]

def search_mmr(query: str, k: int = 6, fetch_k: int = 30, where: Optional[Dict[str, Any]] = None) -> List[Document]:
    return _db().max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, filter=where)
