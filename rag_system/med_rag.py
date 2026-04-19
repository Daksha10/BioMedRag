import json  # Import JSON module for parsing and formatting data
import os  # Import OS module for environment variable access
import time  # Import time module for metrics calculation
from openAI_chat import Chat  # Import default OpenAI chat handler
from bm25_retriever import BM25Retriever  # Import the lexical search retriever
from hybrid_retriever import HybridRetriever  # Import the neural reranker retriever

from dotenv import load_dotenv  # Import dotenv to load .env files
load_dotenv()  # Load environment variables from the local .env file

class MedRAG:
    """
    The main orchestrator class for the Medical RAG system.
    It coordinates document retrieval and LLM response generation.
    """
    def __init__(self, retriever=2, question_type=1, n_docs=10):
        # Initialize the chosen retriever based on the ID passed from the UI
        if retriever == 2:
            self.retriever = BM25Retriever()  # Use standard keyword-based search
        elif retriever == 3:
            self.retriever = HybridRetriever()  # Use keyword search + neural reranking
        elif retriever == 5:
            # Lazy import to avoid loading heavy DPR models unless specifically needed
            from dpr_retriever import DPRRetriever
            self.retriever = DPRRetriever()  # Use semantic vector search
        else:
            # Fallback for invalid input
            raise ValueError(
                "Invalid retriever value. Choose:\n"
                "  2 → BM25 (ES lexical)\n"
                "  3 → Hybrid (BM25 + neural rerank)\n"
                "  5 → DPR (ES dense-vector kNN)"
            )

        # Determine which LLM provider (Gemini, Groq, or OpenAI) to use
        llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()  # Get provider from environment
        if llm_provider == 'gemini':
            from gemini_chat import GeminiChat  # Import Gemini-specific wrapper
            self.chat = GeminiChat(question_type=question_type)  # Instantiate Gemini handler
        elif llm_provider == 'groq':
            from groq_chat import GroqChat  # Import Groq-specific wrapper
            self.chat = GroqChat(question_type=question_type)  # Instantiate Groq handler
        else:
            from openAI_chat import Chat  # Default to standard OpenAI wrapper
            self.chat = Chat(question_type=question_type)  # Instantiate OpenAI handler
            
        self.n_docs = n_docs  # Store the number of documents to retrieve

    def extract_pmids(self, docs):
        """Extracts PMIDs from the retrieved documents and returns them as a list."""
        # Loop through document dictionary and extract the 'PMID' field for each hit
        return [doc["PMID"] for doc in docs.values()]

    def get_answer(self, question: str) -> str:
        """Executes the full RAG pipeline: retrieval followed by generation."""

        # 1. RETRIEVAL STEP
        start_time_retrieval = time.time()  # Start timer for retrieval
        # Call the specific retriever instance to fetch relevant docs from Elasticsearch
        retrieved_docs_json = self.retriever.retrieve_docs(question, self.n_docs)
        retrieved_docs = json.loads(retrieved_docs_json)  # Parse the JSON string into a dict
        end_time_retrieval = time.time()  # End timer for retrieval

        # 2. IDENTIFY SOURCES
        # Extract the source PMIDs to track which documents were available to the LLM
        pmids = self.extract_pmids(retrieved_docs)

        # 3. GENERATION STEP
        start_time_generation = time.time()  # Start timer for LLM generation
        # Send the context documents and user question to the LLM provider
        answer = self.chat.create_chat(question, retrieved_docs)
        end_time_generation = time.time()  # End timer for generation

        # 4. METRICS CALCULATION
        retrieval_time = end_time_retrieval - start_time_retrieval  # Calculate elapsed retrieval time
        generation_time = end_time_generation - start_time_generation  # Calculate elapsed generation time

        # 5. RESPONSE PACKAGING
        try:
            answer_data = json.loads(answer)  # Attempt to parse the LLM's JSON response
            
            # Defensive programming: ensure the expected 'response' key exists
            if "response" not in answer_data:
                answer_data["response"] = str(answer_data)
            # Ensure 'used_PMIDs' list exists for citation tracking
            if "used_PMIDs" not in answer_data:
                answer_data["used_PMIDs"] = []
                
            # Enrich the response with timing metrics and source info
            answer_data["retrieved_PMIDs"] = pmids  # All PMIDs found by retriever
            answer_data["retrieval_time"] = retrieval_time  # Performance metric
            answer_data["generation_time"] = generation_time  # Performance metric
            return json.dumps(answer_data)  # Return final enriched JSON string
            
        except Exception as e:
            # Fallback error handling if LLM fails to return valid JSON
            fallback = {
                "response": f"Error parsing response: {str(e)}",
                "used_PMIDs": [],
                "retrieved_PMIDs": pmids,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "raw_answer": str(answer)
            }
            return json.dumps(fallback)  # Return structure fallback object
