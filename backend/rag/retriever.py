from rag.embeddings import get_chroma_client, get_or_create_collection

def store_chunks(chunks: list):
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    documents = [chunk["text"] for chunk in chunks]
    ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in chunks]
    metadatas = [{"source": chunk["source"], "chunk_id": chunk["chunk_id"]} for chunk in chunks]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    return len(documents)

def retrieve_chunks(query: str, n_results: int = 5) -> list:
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results["documents"][0]