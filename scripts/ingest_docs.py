"""PDF/Markdown を Chroma に投入"""
import sys, pathlib
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import CHROMA_PERSIST_PATH, OPENAI_API_KEY
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

def ingest(path: pathlib.Path):
    loader = PyPDFLoader(str(path)) if path.suffix.lower()==".pdf" else TextLoader(str(path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    db = Chroma(persist_directory=str(CHROMA_PERSIST_PATH),
                embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    db.add_documents(chunks)
    db.persist()
    print(f"Added {len(chunks)} chunks from {path}")

if __name__ == "__main__":
    ingest(pathlib.Path(sys.argv[1]))
