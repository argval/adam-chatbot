import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import chromadb

def query_vector_store(query: str, vector_store_path: str, n_results: int = 5):
    """
    Queries the vector store for relevant chunks.

    Args:
        query (str): The query string.
        vector_store_path (str): Path to the vector database.
        n_results (int): Number of results to return.

    Returns:
        List of relevant documents.
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=vector_store_path)

    # Get collection
    collection = client.get_collection(name="documents")

    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # Format results
    formatted_results = []
    for i in range(len(results['documents'][0])):
        formatted_results.append({
            "content": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "score": results['distances'][0][i] if 'distances' in results else None
        })

    return formatted_results

if __name__ == "__main__":
    # Example query based on your data
    query = "What is the attention mechanism?"
    results = query_vector_store(query, "data/vector_store")
    print(f"Query: {query}\n")
    for idx, res in enumerate(results, start=1):
        score = res.get("score")
        metadata = res.get("metadata", {})
        content = res.get("content", "")
        if isinstance(score, (int, float)):
            print(f"{idx}. score={score:.4f}")
        else:
            print(f"{idx}. score={score}")
        print("Metadata:")
        print(json.dumps(metadata, indent=2))
        preview = content if len(content) <= 400 else content[:400] + "..."
        print("Content:")
        print(preview)
        print("-" * 80)
