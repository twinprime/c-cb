from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
import time
import storage


def query(prompt: str):
    print("Query started")
    start = time.perf_counter()
    response = query_engine.chat(prompt)
    timing = time.perf_counter() - start
    print(response)
    print(f"Query time: {timing:.2f}s")


embedding = storage.load_embedding()

# ollama
llm = Ollama(model="qwen2.5-coder:7b", request_timeout=300.0)

print("Reading index...")
storage_context = storage.get_storage_context("test")
index = VectorStoreIndex.from_vector_store(
    storage_context.vector_store, embed_model=embedding
)
print("Index read")

query_engine = index.as_chat_engine(ChatMode.CONTEXT, llm=llm)

while True:
    p = input("Prompt: ")
    query(p)
