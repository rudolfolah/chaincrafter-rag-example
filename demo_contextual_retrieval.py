import os
from retriever import Retriever
from main import documents as main_documents, queries as main_queries
from model_provider import model_names

# Workaround for unstructured
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# We join them back together to form two larger "documents" (France and England)
# to demonstrate how chunks are situated within their context.
france_text = "\n".join([doc for doc in main_documents if "France" in doc or "French" in doc or "Rhine" in doc or "Mediterranean" in doc][:20])
england_text = "\n".join([doc for doc in main_documents if "England" in doc or "English" in doc or "Great Britain" in doc])
demo_documents = [france_text, england_text]

def run_demo():
    print("--- Demo: Contextual Retrieval using main.py text ---")
    
    # Common settings
    llm_model, embedding_model = model_names("google/gemma-3-4b")
    # Using the first query from main.py
    query = main_queries[0]
    
    # Case 1: Without Contextual Retrieval
    print("\n[Case 1: Without Contextual Retrieval]")
    retriever_basic = Retriever(
        documents=demo_documents,
        embedding_model=embedding_model,
        llm_model=llm_model,
        use_contextual_retrieval=False,
        use_hybrid=False,
        use_reranking=False
    )
    retriever_basic.prepare()
    
    results = retriever_basic.retrieve(query, num_neighbors=1)
    print(f"Query: {query}")
    print(f"Top Result:\n{results[0].text}")

    # Case 2: With Contextual Retrieval
    print("\n[Case 2: With Contextual Retrieval]")
    retriever_context = Retriever(
        documents=demo_documents,
        embedding_model=embedding_model,
        llm_model=llm_model,
        use_contextual_retrieval=True,
        use_hybrid=False,
        use_reranking=False
    )
    retriever_context.prepare()
    
    results = retriever_context.retrieve(query, num_neighbors=1)
    print(f"Query: {query}")
    print(f"Top Result:\n{results[0].text}")

if __name__ == "__main__":
    run_demo()
