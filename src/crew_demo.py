uv run python - <<'PY'
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import chromadb


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

docs = []
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
for file in Path("data/pdfs").glob("*.pdf"):
    pages = PyPDFLoader(str(file)).load()
    docs.extend(text_splitter.split_documents(pages))

client = chromadb.PersistentClient(path="data/chroma")
vectordb = Chroma(
    client=client,
    collection_name="paperbuddy",
    embedding_function=lambda texts: EMBEDDING_MODEL.encode(texts).tolist(),
)
vectordb.add_documents(docs)
print("Ok!")
PY