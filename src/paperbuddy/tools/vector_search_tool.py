from __future__ import annotations
from typing import List
from crewai_tools import tool
from paperbuddy.tools.vector_search import search as vs_search

@tool
def vector_search(query: str, k: int = 5, truncate: int = 1000) -> List[str]:
    """
    Busca trechos relevantes nos PDFs indexados.
    Retorna lista de strings formatadas: "filename (p.X): trecho..."
    truncate: número máximo de caracteres do trecho (0 = sem corte)
    """
    try:
        docs_scores = vs_search(query, k=k)
    except Exception as e:
        return [f"[ERRO] Falha ao buscar: {e}"]

    results = []
    for doc, _ in docs_scores:
        fn = doc.metadata.get("filename", "arquivo_desconhecido")
        pg = doc.metadata.get("page", "?")
        content = doc.page_content.strip()
        if truncate and len(content) > truncate:
            content = content[:truncate] + "..."
        results.append(f"{fn} (p.{pg}): {content}")

    return results
