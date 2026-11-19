# Given a directory of documents, the user can input questions and receive answers with context augmented by retrieved documents
import os
from pathlib import Path

from model_provider import model_names
from retriever import ChunkerKind
from retrieval_augmented_generator import RetrievalAugmentedGenerator

# Workaround needed for unstructured, without this there is an error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

documents = sorted(Path("./docs").glob("*.pdf"))
documents.extend(sorted(Path("./docs").glob("*.md")))
print("Documents found:", documents)
chat_model, embedding_model = model_names("gemma-4-e4b-it")
rag = RetrievalAugmentedGenerator(
    documents=documents,
    queries=[],
    embedding_model=embedding_model,
    llm_model=chat_model,
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
        for answer in rag.generate():
            print(f"--- {answer.query}")
            for message in answer.messages:
                print(f"{message['role']}: {message['content']}")
            print()
except (KeyboardInterrupt, EOFError):
    print("\nExiting...")
