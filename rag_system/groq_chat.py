import openai  # Import the OpenAI Python library for API communication
import os  # Import the OS module to access environment variables
import json  # Import the JSON module for data parsing and serialization
from typing import List, Dict  # Import type hints for better code clarity
from dotenv import load_dotenv  # Import dotenv to manage configuration via .env files

load_dotenv()  # Load key-value pairs from .env file into the environment

class GroqChat:
    """
    LLM generation module specifically for Groq (LPU) models.
    This class leverages the OpenAI-compatible SDK pointing to Groq's high-speed inference endpoint.
    """
    def __init__(self, question_type: int = 1, api_key: str = os.getenv('GROQ_API_KEY'), model: str = os.getenv('GROQ_MODEL', "llama-3.3-70b-versatile")):
        # Store the API key provided in the constructor or environment
        self.api_key = api_key
        # Default to the Llama 3.3 model if no specific model override is found
        self.model = model
        # Initialize the OpenAI client pointing to Groq's specialized API URL
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        # Set the system instruction/context based on the user's selected question type
        self.context = self.set_context(question_type)

    def set_context(self, question_type: int) -> str:
        """Configures the 'System Instruction' for the LLM based on the task type."""
        # The base context enforces scientific rigor and strict grounding in retrieved data
        base_context = (
            "You are a scientific medical assistant designed to synthesize responses "
            "from specific medical documents. Only use the information provided in the "
            "documents to answer questions. The first documents should be the most relevant."
            "Do not use any other information except for the documents provided."
            "When answering questions, you MUST always format your response "
            "as a STRICT JSON object with these EXACT keys: 'response' (string), 'used_PMIDs' (list of strings). "
            "Cite all PMIDs your response is based on in the 'used_PMIDs' field. "
            "Please think step-by-step before answering questions and provide the most accurate response possible."
        )

        # Task-specific instructions for different UI modes
        question_specific_context = {
            1: " Provide a detailed answer to the question in the 'response' field.",
            2: " Your response should only be 'yes', 'no'. If if no relevant documents are found, return 'no_docs_found'.",
            3: " Choose between the given options 1 to 4 and return as 'response' the chosen number. If no relevant documents are found, return the number 5.",
            4: " Respond with keywords and list each keyword separately as a list element. For example ['keyword1', 'keyword2', 'keyword3']. If no relevant documents are found, return an empty list.",
        }

        # Combine base scientific instructions with task-specific output rules
        return base_context + question_specific_context.get(question_type, "")

    def set_initial_message(self) -> List[dict]:
        """Creates the initial system message containing the RAG instructions."""
        # Returns the core persona and rules for the LLM to follow throughout the chat
        return [{"role": "system", "content": self.context}]

    def create_chat(self, user_message: str, retrieved_documents: Dict) -> str:
        """Sends the retrieved context and question to Groq and parses the JSON result."""
        # 1. Start with the system instruction
        messages = self.set_initial_message()
        # 2. Add the actual question from the user
        messages.append({"role": "user", "content": f"Answer the following question: {user_message}"})
        
        # 3. Format the scientific abstracts into a readable text block for the LLM
        document_texts = ["PMID {}: {} {}".format(doc['PMID'], doc['title'], doc['content']) for doc in retrieved_documents.values()]
        # Inject the documents as a 'system' block to clearly separate them from the user prompt
        documents_message = "CONTEXT DOCUMENTS:\n" + "\n\n".join(document_texts)
        messages.append({"role": "system", "content": documents_message})

        try:
            # Execute the API call using Groq's OpenAI-compatible completions endpoint
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.0,  # Temperature 0 for maximum deterministic accuracy
                # Force Groq to output valid JSON to match our system's requirements
                response_format={"type": "json_object"}
            )
            
            # Extract the raw text from the completion
            response_content = completion.choices[0].message.content
            
            # 4. ROBUST JSON PARSING
            # Some models wrap JSON in markdown blocks (```json ... ```); we strip them if present
            raw_json = response_content
            if "```json" in raw_json:
                raw_json = raw_json.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_json:
                raw_json = raw_json.split("```")[1].split("```")[0].strip()
            
            try:
                # Parse the sanitized string into a Python dictionary
                response_data = json.loads(raw_json)
                # Map potential alternate keys (like 'answer') to our standard 'response' key
                final_data = {
                    "response": response_data.get("response") or response_data.get("answer") or raw_json,
                    "used_PMIDs": response_data.get("used_PMIDs", []),
                    "retrieved_PMIDs": [doc['PMID'] for doc in retrieved_documents.values()]
                }
                # Serialize back to a clean JSON string for the MedRAG orchestrator
                return json.dumps(final_data)
            except Exception as e:
                # If JSON parsing still fails, return the raw text inside our expected schema
                return json.dumps({
                    "response": response_content,
                    "used_PMIDs": [],
                    "retrieved_PMIDs": [doc['PMID'] for doc in retrieved_documents.values()],
                    "debug_error": f"JSONDecodeError: {str(e)}"
                })
        
        except Exception as e:
            # Catch network or API-level errors
            return json.dumps({"error": str(e)})
