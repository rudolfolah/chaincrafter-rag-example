import os
from retriever import Retriever
from main import documents as main_documents, queries as main_queries

# Workaround for unstructured
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_demo():
    print("--- Demo: Re-ranking using main.py text ---")
    
    # Common settings
    embedding_model = "text-embedding-nomic-embed-text-v1.5"
    # Using the first query from main.py
    query = main_queries[0]
    
    # Case 1: Without Re-ranking
    print("\n[Case 1: Without Re-ranking]")
    retriever_no_rerank = Retriever(
        documents=main_documents,
        embedding_model=embedding_model,
        use_hybrid=False,
        use_reranking=False,
        use_contextual_retrieval=False
    )
    retriever_no_rerank.prepare()
    results = retriever_no_rerank.retrieve(query, num_neighbors=2)
    print(f"Query: {query}")
    print("Top 2 Results (No Rerank):")
    for i, res in enumerate(results):
        print(f"[{i+1}] {res.text}")

    # Case 2: With Re-ranking (Cross-encoder)
    print("\n[Case 2: With Re-ranking]")
    retriever_rerank = Retriever(
        documents=main_documents,
        embedding_model=embedding_model,
        use_hybrid=False,
        use_reranking=True,
        use_contextual_retrieval=False
    )
    retriever_rerank.prepare()
    results = retriever_rerank.retrieve(query, num_neighbors=2)
    print(f"Query: {query}")
    print("Top 2 Results (With Rerank):")
    for i, res in enumerate(results):
        print(f"[{i+1}] {res.text}")

if __name__ == "__main__":
    run_demo()
