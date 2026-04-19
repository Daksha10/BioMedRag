from elasticsearch import Elasticsearch  # Import the official Elasticsearch client for document retrieval
import os  # Import the OS module for reading environment variables
import json  # Import the JSON module for structuring response data
from medCPT_encoder import MedCPTCrossEncoder  # Import the neural cross-encoder for semantic reranking
from dotenv import load_dotenv  # Import dotenv to load secret keys and configurations from .env

load_dotenv()  # Initialize the environment by loading the .env file

class HybridRetriever:
    """
    Implements a two-stage retrieval pipeline:
    1. Lexical Search: Fetch candidate documents using BM25 keyword matching via Elasticsearch.
    2. Neural Reranking: Use MedCPTCrossEncoder to re-order candidates by semantic relevance to the query.
    """
    def __init__(self):
        # ── ELASTICSEARCH CONFIGURATION ──────────────────────────────────────────
        elastic_password = os.getenv('ELASTIC_PASSWORD')
        elastic_user = os.getenv('ELASTIC_USER', 'elastic')
        es_host = os.getenv('ES_HOST', 'https://localhost:9200')
        cert_path = os.getenv('ES_CERT_PATH')
        
        # ── CLIENT INITIALIZATION ───────────────────────────────────────────────
        if cert_path and os.path.exists(cert_path):
            # Setup secure connection with CA certificate verification
            self.es = Elasticsearch(
                [es_host],
                basic_auth=(elastic_user, elastic_password),
                verify_certs=True,
                ca_certs=cert_path,
                request_timeout=60
            )
        else:
            # Setup connection for development environments (skipping SSL verification)
            self.es = Elasticsearch(
                [es_host],
                basic_auth=(elastic_user, elastic_password),
                verify_certs=False,
                request_timeout=60
            )
        # Store the target index (e.g., 'pubmed_index')
        self.index = os.getenv('INDEX_NAME', 'pubmed_index')
        # Initialize the MedCPT Cross-Encoder (this model compares query/doc pairs directly)
        self.reranker = MedCPTCrossEncoder()

    def rerank_docs(self, query: str, docs: list):
        """
        Uses the MedCPT Cross-Encoder to compute relevance scores for a list of documents.
        Returns the documents sorted from highest to lowest semantic score.
        """
        # Calculate cross-attention scores for all retrieved document contents against the query
        scores = self.reranker.score([doc['content'] for doc in docs], query)
        # Pair documents with their scores and sort them in descending order
        reranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return reranked_docs

    def retrieve_docs(self, query: str, top_n: int = 10, k: int = 20):
        """
        Executes the full hybrid search: retrieves 'k' candidates and returns the best 'top_n'.
        """
        # ── STAGE 1: KEYWORD RETRIEVAL (BM25) ────────────────────────────────────
        # Fetch a larger pool of candidates (k) than we ultimately need (top_n)
        es_query = {
            "size": k,
            "query": {
                "match": {
                    "content": query  # Lexical match on the abstract text
                }
            },
            "_source": ["PMID", "title", "content"]
        }
        
        # Execute search in Elasticsearch
        response = self.es.search(index=self.index, body=es_query)

        # Extract document data into a list of dictionaries
        docs = [{
            'PMID': hit['_source']['PMID'],
            'title': hit['_source']['title'],
            'content': hit['_source']['content']
        } for hit in response['hits']['hits']]

        # ── STAGE 2: NEURAL RERANKING ───────────────────────────────────────────
        # Apply the deep learning reranker to improve the precision of the top results
        reranked_docs = self.rerank_docs(query, docs)

        # Filter: only keep documents that the reranker gave a positive score
        reranked_docs = [doc for doc in reranked_docs if doc[1] > 0]

        # Select exactly the requested number of top documents
        top_reranked_docs = reranked_docs[:top_n]

        # ── STAGE 3: OUTPUT PACKAGING ───────────────────────────────────────────
        results = {
            f"doc{idx + 1}": {
                'PMID': doc['PMID'],
                'title': doc['title'],
                'content': doc['content'],
                'score': score.item()  # Include the neural relevance score
            }
            for idx, (doc, score) in enumerate(top_reranked_docs)
        }

        # Return the final results in a structured JSON string
        return json.dumps(results, indent=4)

if __name__ == "__main__":
    # Integration test for the HybridRetriever class
    retriever = HybridRetriever()
    query = "Is Alzheimer's disease hereditary?"
    results = retriever.retrieve_docs(query, k=100, top_n=10)
    print(results)
