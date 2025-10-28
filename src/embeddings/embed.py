import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import chromadb
from langchain_openai import OpenAIEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)

def embed_and_store(processed_file: str, vector_store_path: str):
    """
    Loads chunks from processed JSON, generates embeddings, and stores in ChromaDB.

    Args:
        processed_file (str): Path to documents.json.
        vector_store_path (str): Path to store the vector database.
    """
    # Load processed documents
    with open(processed_file, 'r') as f:
        docs_data = json.load(f)

    if not docs_data:
        logging.warning("No documents found in processed file.")
        return

    # Initialize embedding model
    model = OpenAIEmbeddings()

    # Prepare data
    documents = [doc['page_content'] for doc in docs_data]
    metadatas = [doc['metadata'] for doc in docs_data]
    ids = [f"chunk_{i}" for i in range(len(documents))]

    # Generate embeddings
    embeddings = model.embed_documents(documents)
    logging.info(f"Generated embeddings for {len(documents)} chunks.")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=vector_store_path)

    # Create or get collection
    collection = client.get_or_create_collection(name="documents")

    # Add documents to collection
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    logging.info(f"Stored {len(documents)} chunks in vector store at {vector_store_path}")

if __name__ == "__main__":
    embed_and_store("data/processed/documents.json", "data/vector_store")
