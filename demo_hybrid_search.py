import os
from retriever import Retriever
from main import documents as main_documents, queries as main_queries

# Workaround for unstructured
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_demo():
    print("--- Demo: Hybrid Search (Vector + BM25 with RRF) using main.py text ---")
    
    # Common settings
    embedding_model = "text-embedding-nomic-embed-text-v1.5"
    # Using the fourth query from main.py
    query = main_queries[3]
    
    # Case 1: Vector Only
    print("\n[Case 1: Vector Only]")
    retriever_vector = Retriever(
        documents=main_documents,
        embedding_model=embedding_model,
        use_hybrid=False,
        use_reranking=False,
        use_contextual_retrieval=False
    )
    retriever_vector.prepare()
    results = retriever_vector.retrieve(query, num_neighbors=1)
    print(f"Query: {query}")
    print(f"Top Result: {results[0]}")

    # Case 2: Hybrid (Vector + BM25 with RRF)
    print("\n[Case 2: Hybrid (Vector + BM25 with RRF)]")
    retriever_hybrid = Retriever(
        documents=main_documents,
        embedding_model=embedding_model,
        use_hybrid=True,
        use_reranking=False,
        use_contextual_retrieval=False
    )
    retriever_hybrid.prepare()
    results = retriever_hybrid.retrieve(query, num_neighbors=1)
    print(f"Query: {query}")
    print(f"Top Result: {results[0]}")

if __name__ == "__main__":
    run_demo()
