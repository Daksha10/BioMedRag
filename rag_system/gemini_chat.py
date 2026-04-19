from google import genai
from google.genai import types
import os
import json
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class GeminiChat:
    def __init__(self, question_type: int = 1, api_key: str = os.getenv('GEMINI_API_KEY'), model: str = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')):
        self.api_key = api_key
        self.model_name = model
        self.client = genai.Client(api_key=self.api_key)
        self.context = self.set_context(question_type)

    def set_context(self, question_type: int) -> str:
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

        question_specific_context = {
            1: " Provide a detailed answer to the question in the 'response' field.",
            2: " Your response should only be 'yes', 'no'. If if no relevant documents are found, return 'no_docs_found'.",
            3: " Choose between the given options 1 to 4 and return as 'response' the chosen number. If no relevant documents are found, return the number 5.",
            4: " Respond with keywords and list each keyword separately as a list element. For example ['keyword1', 'keyword2', 'keyword3']. If no relevant documents are found, return an empty list.",
        }

        return base_context + question_specific_context.get(question_type, "")

    def create_chat(self, user_message: str, retrieved_documents: Dict) -> str:
        document_texts = ["PMID {}: {} {}".format(doc['PMID'], doc['title'], doc['content']) for doc in retrieved_documents.values()]
        documents_context = "CONTEXT DOCUMENTS:\n" + "\n\n".join(document_texts)
        
        full_prompt = f"{documents_context}\n\nUSER QUESTION: {user_message}"

        try:
            # Generate content using the NEW google-genai SDK
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.context,
                    temperature=0.0,
                    max_output_tokens=1000,
                    response_mime_type="application/json"
                )
            )
            
            response_content = response.text.strip()
            # Clean up potential markdown formatting
            if response_content.startswith("```json"):
                response_content = response_content.replace("```json", "", 1)
            if response_content.endswith("```"):
                response_content = response_content.rsplit("```", 1)[0]
            response_content = response_content.strip()

            try:
                response_data = json.loads(response_content)
                
                # Check for double-wrapping (common in some Flash models)
                if isinstance(response_data, dict) and "response" in response_data and isinstance(response_data["response"], dict):
                    # It's double wrapped, let's flatten it
                    inner = response_data["response"]
                    response_data["response"] = inner.get("response") or inner.get("answer")
                    if not response_data.get("used_PMIDs"):
                        response_data["used_PMIDs"] = inner.get("used_PMIDs", [])

                # Ensure keys exist even if model was lazy (though response_mime_type helps)
                answer_text = response_data.get("response") or response_data.get("answer") or response_content
                if isinstance(answer_text, dict):
                    # Still a dict? Just take its values
                    answer_text = str(answer_text)

                final_data = {
                    "response": answer_text,
                    "used_PMIDs": response_data.get("used_PMIDs", []),
                    "retrieved_PMIDs": [doc['PMID'] for doc in retrieved_documents.values()]
                }
                return json.dumps(final_data)
            except Exception as e:
                # Fallback if somehow JSON parsing fails
                return json.dumps({
                    "response": response_content,
                    "used_PMIDs": [],
                    "retrieved_PMIDs": [doc['PMID'] for doc in retrieved_documents.values()],
                    "debug_error": f"JSONDecodeError: {str(e)}"
                })
        
        except Exception as e:
            return json.dumps({"error": str(e)})
