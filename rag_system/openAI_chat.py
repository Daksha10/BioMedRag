import openai
import os
import json
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class Chat:
    def __init__(self, question_type: int = 1, api_key: str = os.getenv('OPENAI_API_KEY'), model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
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
            4: " Respond with keywords and list each keyword sepeartly as a list element. For example ['keyword1', 'keyword2', 'keyword3']. If no relevant documents are found, return an empty list.",
        }

        return base_context + question_specific_context.get(question_type, "")

    def set_initial_message(self) -> List[dict]:
        return [{"role": "system", "content": self.context}]

    def create_chat(self, user_message: str, retrieved_documents: Dict) -> str:
        messages = self.set_initial_message()
        messages.append({"role": "user", "content": f"Answer the following question: {user_message}"})
        
        document_texts = ["PMID {}: {} {}".format(doc['PMID'], doc['title'], doc['content']) for doc in retrieved_documents.values()]
        documents_message = "\n\n".join(document_texts)  # Separating documents with two newlines
        messages.append({"role": "system", "content": documents_message})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.0
            )
            
            response_content = completion.choices[0].message.content
            
            # Robust JSON parsing: handle markdown blocks if present
            raw_json = response_content
            if "```json" in raw_json:
                raw_json = raw_json.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_json:
                raw_json = raw_json.split("```")[1].split("```")[0].strip()
            
            try:
                response_data = json.loads(raw_json)
                # Ensure keys exist even if model was lazy
                final_data = {
                    "response": response_data.get("response") or response_data.get("answer") or raw_json,
                    "used_PMIDs": response_data.get("used_PMIDs", []),
                    "retrieved_PMIDs": [doc['PMID'] for doc in retrieved_documents.values()]
                }
                return json.dumps(final_data)
            except Exception as e:
                # If JSON fails, return the raw content in a structured way for debugging
                return json.dumps({
                    "response": response_content,
                    "used_PMIDs": [],
                    "retrieved_PMIDs": [doc['PMID'] for doc in retrieved_documents.values()],
                    "debug_error": f"JSONDecodeError: {str(e)}"
                })
        
        except Exception as e:
            return json.dumps({"error": str(e)})