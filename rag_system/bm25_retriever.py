from elasticsearch import Elasticsearch  # Import the official Elasticsearch client
import os  # Import the OS module to read environment configurations
import json  # Import JSON for structuring retrieved metadata
from dotenv import load_dotenv  # Import dotenv for easy variable management via .env

load_dotenv()  # Load variables from the .env file in the project root

class BM25Retriever:
    """
    Implements a lexical search retriever using the BM25 algorithm via Elasticsearch.
    BM25 (Best Matching 25) is the industry standard for keyword-based document retrieval.
    """
    def __init__(self):
        # ── CONFIGURATION ────────────────────────────────────────────────────────
        # Retrieve credentials and connection details from environment variables
        elastic_password = os.getenv('ELASTIC_PASSWORD')
        elastic_user = os.getenv('ELASTIC_USER', 'elastic')
        es_host = os.getenv('ES_HOST', 'https://localhost:9200')
        cert_path = os.getenv('ES_CERT_PATH')
        
        # ── CLIENT INITIALIZATION ───────────────────────────────────────────────
        # Setup connection with security checks based on available certificates
        if cert_path and os.path.exists(cert_path):
            self.es = Elasticsearch(
                [es_host],
                basic_auth=(elastic_user, elastic_password),
                verify_certs=True,  # Secure production-style verification
                ca_certs=cert_path,  # Path to the CA certificate
                request_timeout=60
            )
        else:
            # Fallback for local development or Docker containers without custom SSL
            self.es = Elasticsearch(
                [es_host],
                basic_auth=(elastic_user, elastic_password),
                verify_certs=False,  # Allow self-signed certificates for dev
                request_timeout=60
            )
        # Store the target index name (e.g., 'pubmed_index')
        self.index = os.getenv('INDEX_NAME', 'pubmed_index')

    def retrieve_docs(self, query: str, k: int = 10):
        """
        Queries Elasticsearch using the BM25 lexical search algorithm.
        Returns a JSON string containing the top K most relevant documents.
        """
        # Define the Elasticsearch search body
        es_query = {
            "size": k,  # Limit results to K documents
            "query": {
                "match": {
                    "content": query  # Core BM25 search on the 'content' field
                }
            },
            # Only return specific fields to save network bandwidth and memory
            "_source": ["PMID", "title", "content"]
        }
        
        # ── EXECUTION ────────────────────────────────────────────────────────────
        # Perform the search against the specified Elasticsearch index
        response = self.es.search(index=self.index, body=es_query)
        
        # ── POST-PROCESSING ──────────────────────────────────────────────────────
        results = {}
        # Iterate through hits and restructure into our standardized RAG format
        for idx, doc in enumerate(response['hits']['hits'], 1):
            doc_key = f"doc{idx}"  # Create keys like 'doc1', 'doc2', etc.
            results[doc_key] = {
                'PMID': doc['_source']['PMID'],  # Use the PubMed ID for citation
                'title': doc['_source']['title'],  # Document title for context
                'content': doc['_source']['content'],  # The actual abstract text
                'score': doc['_score']  # Keep the BM25 relevance score for transparency
            }

        # Return the results as a human-readable JSON string
        return json.dumps(results, indent=4)