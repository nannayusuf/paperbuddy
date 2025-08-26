from __future__ import annotations
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

ROOT = Path(__file__).resolve().parents[3]
PDF_DIR = ROOT / "data" / "pdfs"
CHROMA_DIR = ROOT / "data" / "chroma"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "paperbuddy"

def _iter_pdfs(pdf_dir: Path) -> Iterable[Path]:
    return sorted(pdf_dir.glob("*.pdf"))

def build_vectorstore(pdf_dir: Path | str = PDF_DIR, chroma_dir: Path | str = CHROMA_DIR) -> None:
    """Indexa todos os PDFs encontrados em `pdf_dir`."""
    pdf_dir = Path(pdf_dir)
    chroma_dir = Path(chroma_dir)

    if not pdf_dir.exists():
        raise FileNotFoundError(str(pdf_dir.absolute()))
    pdfs = list(_iter_pdfs(pdf_dir))
    if not pdfs:
        raise FileNotFoundError(str(pdf_dir.absolute()))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    docs = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"[paperbuddy] ({i}/{len(pdfs)}) → {pdf.name}")
        pages = PyPDFLoader(str(pdf)).load()
        docs.extend(splitter.split_documents(pages))

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Limpa coleção antiga
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    vectordb = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    if docs:
        vectordb.add_documents(docs)

    print(f"[paperbuddy] PDFs: {len(pdfs)}")
    print(f"[paperbuddy] Chunks: {len(docs)}")
    print(f"[paperbuddy] Coleção: {COLLECTION_NAME} em {chroma_dir}")

def main():
    build_vectorstore()

if __name__ == "__main__":
    main()