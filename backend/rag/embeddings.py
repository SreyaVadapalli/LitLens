import chromadb
from chromadb.utils import embedding_functions

def get_chroma_client():
    client = chromadb.PersistentClient(path="../chroma_db")
    return client

def get_or_create_collection(client, collection_name="litlens_papers"):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="allenai/specter"
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef
    )
    return collection