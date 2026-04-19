import json
import os
from dotenv import load_dotenv

# Import the RAG system
import sys
sys.path.append(os.path.join(os.getcwd(), 'rag_system'))
from med_rag import MedRAG

def run_smoke_test():
    load_dotenv()
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY') or 'your_openai_api_key' in os.getenv('OPENAI_API_KEY'):
        print("Error: Please set your OPENAI_API_KEY in the .env file.")
        return

    # Initialize the RAG system
    # 2 for BM25 (easiest to test first)
    # 1 for Full text response
    try:
        provider = os.getenv('LLM_PROVIDER', 'openai').upper()
        print(f"\n--- Medical RAG Smoke Test (LLM: {provider}) ---")
        print("Initializing MedRAG with BM25 retriever...")
        rag = MedRAG(retriever=2, question_type=1, n_docs=3)
        
        question = "What are the structural proteins of a coronavirus?"
        print(f"\nQuestion: {question}")
        
        print("Getting answer from RAG system (this may take a moment)...")
        response_json = rag.get_answer(question)
        
        if response_json:
            print(f"\nRAW JSON OUTPUT: {response_json}")
            response = json.loads(response_json)
            print("\n--- Response ---")
            print(response.get('response'))
            print(f"\nUsed PMIDs: {response.get('used_PMIDs')}")
            print(f"Retrieved PMIDs: {response.get('retrieved_PMIDs')}")
            print(f"Retrieval Time: {response.get('retrieval_time', 0):.2f}s")
            print(f"Generation Time: {response.get('generation_time', 0):.2f}s")
        else:
            print("\nResponse was empty. Check if Elasticsearch is running and documents are ingested.")
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Hint: Make sure Elasticsearch is running on https://localhost:9200 and you have ingested sample data.")

if __name__ == "__main__":
    run_smoke_test()
