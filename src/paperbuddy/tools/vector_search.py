import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("papers")
model = SentenceTransformer('BAAI/bge-m3')

def vector_search_tool(query: str, filter_type: str = None) -> list:
    """Ferramenta de busca vetorial para agentes"""
    emb = model.encode(query).tolist()
    
    where = {"type": filter_type} if filter_type else None
    
    results = collection.query(
        query_embeddings=[emb],
        n_results=5,
        where=where
    )
    
    return [{"content": doc, "metadata": meta} 
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])]