from fastapi import FastAPI
from paperbuddy.agents.coordinator import process_query

app = FastAPI()

@app.post("/ask")
async def ask_question(query: str, paper_id: str):
    result = process_query(query, paper_id)
    return {"answer": result, "agents_involved": ["text", "vision", "rag"]}

@app.post("/ingest")
async def ingest_paper(pdf_path: str, paper_id: str):
    from paperbuddy.indexing.ingest import ingest_paper
    ingest_paper(pdf_path, paper_id)
    return {"status": "indexed", "paper_id": paper_id}