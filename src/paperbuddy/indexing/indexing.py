import chromadb
from paperbuddy.tools.pdf_parse import parse_pdf
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("papers")

model = SentenceTransformer('BAAI/bge-m3')

def ingest_paper(pdf_path: str, paper_id: str):
    """Indexa paper no ChromaDB"""
    parsed = parse_pdf(pdf_path)
    
    # Indexar texto
    chunks = parsed["text"].split("\n\n")
    embeddings = model.encode(chunks)
    
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(
            embeddings=[emb.tolist()],
            documents=[chunk],
            metadatas=[{"type": "text", "paper_id": paper_id, "chunk_id": i}]
        )
    
    # Indexar tabelas
    for i, table in enumerate(parsed["tables"]):
        table_emb = model.encode(table)
        collection.add(
            embeddings=[table_emb.tolist()],
            documents=[table],
            metadatas=[{"type": "table", "paper_id": paper_id, "table_id": i}]
        )