from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser import SimpleFileNodeParser, CodeSplitter
import time
from typing import Dict
import os
import storage
import tree_sitter_kotlin
from tree_sitter import Language, Parser


parsers: Dict[str, NodeParser] = {
    "kt": CodeSplitter(
        language="kotlin",
        parser=Parser(Language(tree_sitter_kotlin.language())),
        chunk_lines=40,  # lines per chunk
        chunk_lines_overlap=15,  # lines overlap between chunks
        max_chars=1500,  # max chars per chunk
    ),
    "md": SimpleFileNodeParser(),
}

documents = SimpleDirectoryReader(
    "/Users/yan-shen.lee.e/Developer/technical-challenge",
    recursive=True,
).load_data()
nodes = []
for d in documents:
    fileName = d.metadata.get("file_name")
    ext = os.path.splitext(fileName)[1][1:]
    parser = parsers.get(ext)
    if parser is not None:
        print(f"Parsing {d.metadata['file_name']}")
        nodes.extend(parser.get_nodes_from_documents([d]))

storage_context = storage.create_storage_context("test")
embedding = storage.load_embedding()

print("Building index...")
start = time.perf_counter()
index = VectorStoreIndex(nodes, embed_model=embedding, storage_context=storage_context)
timing = time.perf_counter() - start
print(f"Index built ({timing:.2f}s)")
