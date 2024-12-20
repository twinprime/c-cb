import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import Settings


def create_storage_context(coollection_name: str):
    chroma_client = chromadb.PersistentClient(path="db")
    if coollection_name in map(lambda x: x.name, chroma_client.list_collections()):
        chroma_client.delete_collection(coollection_name)
    chroma_collection = chroma_client.create_collection(coollection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return StorageContext.from_defaults(vector_store=vector_store)


def get_storage_context(coollection_name: str):
    chroma_client = chromadb.PersistentClient(path="db")
    chroma_collection = chroma_client.get_collection(coollection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return StorageContext.from_defaults(vector_store=vector_store)


def load_embedding():
    # bge embedding model
    print("Loading embedding model...")
    model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    print("Embedding model loaded")
    return model
