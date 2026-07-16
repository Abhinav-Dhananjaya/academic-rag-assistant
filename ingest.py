
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

# 1. Load your PDFs
print("Loading PDFs...")
loader = PyPDFDirectoryLoader("./data")
documents = loader.load()

# 2. Split into chunks
print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 3. Initialize Embeddings
# Using nomic-embed-text via Ollama
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)

# 4. Store in Qdrant
# We use 'path' for local disk persistence
print("Storing in Vector DB (this may take a moment)...")
vector_db = QdrantVectorStore.from_documents(
    chunks,
    embeddings,
    path="./local_qdrant",
    collection_name="academic_docs"
)

print("Ingestion complete!")
