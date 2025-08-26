from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import argparse, hashlib, shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

ROOT = Path(__file__).resolve().parents[3]
PDF_DIR = ROOT / "data" / "pdfs"
CHROMA_DIR = ROOT / "data" / "chroma"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "paperbuddy"

def _iter_pdfs(pdf_dir: Path) -> Iterable[Path]:
    return sorted(pdf_dir.rglob("*.pdf"))

def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _chunk_docs(pdf: Path) -> List[Document]:
    loader = PyPDFLoader(str(pdf))
    pages = loader.load()
    pages = [p for p in pages if p.page_content and p.page_content.strip()]
    if not pages:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    out: List[Document] = []
    for d in chunks:
        if not d.page_content or not d.page_content.strip():
            continue
        meta = dict(d.metadata or {})
        meta.update({"source": str(pdf), "filename": pdf.name})
        out.append(Document(page_content=d.page_content, metadata=meta))
    return out

def _doc_ids(docs: List[Document]) -> List[str]:
    return [
        _hash_id(f"{d.metadata.get('source','')}:{d.metadata.get('page','')}:{i}:{len(d.page_content)}")
        for i, d in enumerate(docs)
    ]

def build_vectorstore(
    pdf_dir: Path | str = PDF_DIR,
    chroma_dir: Path | str = CHROMA_DIR,
    *, reset: bool = True
) -> Chroma:
    pdf_dir = Path(pdf_dir)
    chroma_dir = Path(chroma_dir)

    if not pdf_dir.exists():
        raise FileNotFoundError(str(pdf_dir.absolute()))
    pdfs = list(_iter_pdfs(pdf_dir))
    if not pdfs:
        raise FileNotFoundError(str(pdf_dir.absolute()))

    if reset and chroma_dir.exists():
        shutil.rmtree(chroma_dir, ignore_errors=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )

    total_docs = 0
    for i, pdf in enumerate(pdfs, 1):
        try:
            docs = _chunk_docs(pdf)
            if not docs:
                print(f"[paperbuddy] ({i}/{len(pdfs)}) {pdf.name} -> 0 chunks")
                continue
            ids = _doc_ids(docs)
            texts = [d.page_content for d in docs]
            metas = []
            for d in docs:
                m = dict(d.metadata or {})
                if not m.get("filename") and m.get("source"):
                    m["filename"] = Path(m["source"]).name
                metas.append(m)
            vectordb.add_texts(texts=texts, metadatas=metas, ids=ids)
            total_docs += len(docs)
            print(f"[paperbuddy] ({i}/{len(pdfs)}) {pdf.name} -> {len(docs)} chunks")
        except Exception as e:
            print(f"[paperbuddy] ERRO em {pdf.name}: {e}")

    collection_count = vectordb._collection.count()
    print(f"[paperbuddy] PDFs: {len(pdfs)}")
    print(f"[paperbuddy] Chunks adicionados: {total_docs}")
    print(f"[paperbuddy] Chunks na coleção agora: {collection_count}")
    print(f"[paperbuddy] Coleção: {COLLECTION_NAME} em {chroma_dir}")
    return vectordb

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    p.add_argument("--db-dir", type=Path, default=CHROMA_DIR)
    p.add_argument("--no-reset", dest="reset", action="store_false")
    p.set_defaults(reset=True)
    args = p.parse_args()
    build_vectorstore(pdf_dir=args.pdf_dir, chroma_dir=args.db_dir, reset=args.reset)

if __name__ == "__main__":
    main()
