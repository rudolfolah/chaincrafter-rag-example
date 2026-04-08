# Given a directory of documents, the user can input questions and receive answers with context augmented by retrieved documents
from pathlib import Path

from retriever import ChunkerKind
from retrieval_augmented_generator import RetrievalAugmentedGenerator

# Workaround needed for unstructured, without this there is an error
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

documents = sorted(Path("./docs").glob("*.pdf"))
documents.extend(sorted(Path("./docs").glob("*.md")))
print("Documents found:", documents)
rag = RetrievalAugmentedGenerator(
    documents=documents,
    queries=[],
    embedding_model="text-embedding-nomic-embed-text-v1.5",
    llm_model="gemma-4-e4b-it",
    temperature=0.7,
    num_neighbors=5,
    chunker=ChunkerKind.UNSTRUCTURED,
)
rag.prepare()

try:
    while True:
        query = input("Enter your question (Ctrl+C or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        rag.queries = [query]
        rag.generate()
except (KeyboardInterrupt, EOFError):
    print("\nExiting...")
