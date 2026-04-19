import os
import sys
import json
from dotenv import load_dotenv

# Path setup to allow importing from rag_system
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAG_SYSTEM_PATH = os.path.join(PROJECT_ROOT, "rag_system")
if RAG_SYSTEM_PATH not in sys.path:
    sys.path.insert(0, RAG_SYSTEM_PATH)

from med_rag import MedRAG

load_dotenv()

def test_groq_integration():
    print("--- Groq Integration Smoke Test ---")
    
    provider = os.getenv("LLM_PROVIDER")
    if provider != "groq":
        print(f"SKIPPING: LLM_PROVIDER is set to '{provider}', not 'groq'.")
        return

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key.startswith("your_"):
        print("ERROR: GROQ_API_KEY not found or is still the placeholder in .env.")
        return

    print(f"Initialising MedRAG with Groq provider (Model: {os.getenv('GROQ_MODEL', 'DEFAULT')})...")
    
    # We use BM25 as it's the fastest for a smoke test
    try:
        rag = MedRAG(retriever=2, n_docs=2)
    except Exception as e:
        print(f"FAILED to initialise MedRAG: {e}")
        return

    # Mock a question
    question = "What are the common symptoms of influenza?"
    print(f"Testing question: '{question}'")
    
    try:
        response_json = rag.get_answer(question)
        response = json.loads(response_json)
        
        print("\n--- Response Received ---")
        # Handle cases where response might be an error dict
        if "error" in response:
            print(f"API Error Detected: {response['error']}")
        else:
            print(f"Answer: {response.get('response')[:200]}...")
            print(f"Used PMIDs: {response.get('used_PMIDs')}")
            print(f"Generation Time: {response.get('generation_time', 'N/A')}s")
            print("\nSUCCESS: Groq integration works and returns the correct JSON schema!")
            
    except Exception as e:
        print(f"\nFAILED to get answer from Groq: {e}")

if __name__ == "__main__":
    test_groq_integration()
