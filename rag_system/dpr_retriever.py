"""
DPR Retriever
--------------
Implements semantic search using the Dense Passage Retrieval (DPR) framework.
Unlike BM25, which looks for exact keyword matches, DPR compares the "meaning" 
of the query to the "meaning" of the document in a high-dimensional vector space.

Flow
----
1. Query Input → The user asks a biomedical question.
2. Encoding → The DPRQueryEncoder turns the text into a 768-dimensional vector.
3. ANN Search → Elasticsearch performs an Approximate Nearest Neighbor search 
   to find the vectors (documents) closest to the query vector using Cosine Similarity.
4. Output → Standardized JSON response with the top-K relevant PubMed abstracts.
"""

from elasticsearch import Elasticsearch  # Import the official Elasticsearch Python client
import numpy as np  # Import numpy for vector math and normalization
import os  # Import the OS module for environment variable access
import json  # Import JSON for structured response handling
from dotenv import load_dotenv  # Import dotenv to load project configurations

load_dotenv()  # Load key-value pairs from .env into environment variables

# Constants for the specialized vector index
DPR_INDEX_NAME = "pubmed_dpr_index"  # Name of the index containing pre-calculated DPR vectors
DPR_VECTOR_FIELD = "dpr_vector"  # The field in ES where the 768-d vectors are stored


def _build_es_client() -> Elasticsearch:
    """
    Helper function to build an Elasticsearch client from environment variables.
    Ensures consistent connection settings (security, timeouts) across all retrievers.
    """
    elastic_password = os.getenv("ELASTIC_PASSWORD")
    elastic_user = os.getenv("ELASTIC_USER", "elastic")
    es_host = os.getenv("ES_HOST", "http://localhost:9200")
    cert_path = os.getenv("ES_CERT_PATH", "")

    # Configure secure connection if a CA certificate path is provided in .env
    if cert_path and os.path.exists(cert_path):
        return Elasticsearch(
            [es_host],
            basic_auth=(elastic_user, elastic_password),
            verify_certs=True,
            ca_certs=cert_path,
            request_timeout=60,
        )
    else:
        # Fallback for standard development setups or local Docker environments
        return Elasticsearch(
            [es_host],
            basic_auth=(elastic_user, elastic_password),
            verify_certs=False,
            request_timeout=60,
        )


class DPRRetriever:
    """
    State-of-the-art retriever utilizing Facebook DPR question encoders 
    and Elasticsearch 8.x kNN (k-Nearest Neighbors) search functionality.
    """

    def __init__(self):
        # Establish connection to the Elasticsearch cluster
        self.es = _build_es_client()
        # Verify that the required vector index has been created
        self._ensure_index_exists()

        # ── LAZY MODEL LOADING ───────────────────────────────────────────────────
        # Import and load the heavy DPR model only when this retriever is instantiated.
        # This prevents slowdowns when starting the app with BM25/Hybrid selected.
        from dpr_encoder import DPRQueryEncoder
        self.query_encoder = DPRQueryEncoder()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_index_exists(self):
        """Validates that the specialized DPR index exists, providing setup help if not."""
        if not self.es.indices.exists(index=DPR_INDEX_NAME):
            # Throw a helpful error directing the user to the encoding script
            raise RuntimeError(
                f"The DPR Elasticsearch index '{DPR_INDEX_NAME}' does not exist.\n"
                "Please build it first by running:\n\n"
                "  python information_retrieval/document_encoding/encode_documents_dpr.py\n\n"
                "This encodes all documents with the DPR passage encoder and stores "
                "the vectors in Elasticsearch. It only needs to be run once."
            )

    def _encode_query(self, query: str) -> list[float]:
        """
        Processes the input query string into a normalized 768-D vector.
        Elasticsearch kNN works best with L2-normalized vectors (unit length).
        """
        # Call the transformer encoder model
        vec = self.query_encoder.encode(query)
        # Normalize the vector to unit length so that Dot Product equals Cosine Similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        # Convert from numpy array to Python list for Elasticsearch compatibility
        return vec.tolist()

    # ------------------------------------------------------------------
    # Public API  (consistent signature used by MedRAG orchestrator)
    # ------------------------------------------------------------------

    def retrieve_docs(self, query: str, k: int = 10) -> str:
        """
        Retrieves the top-k documents semantically related to the query.
        Returns a JSON structure compliant with the RAG generator's schema.
        """
        # 1. Transform the natural language question into a vector
        query_vector = self._encode_query(query)

        # 2. Build the Elasticsearch 8.x kNN search body
        es_body = {
            "knn": {
                "field": DPR_VECTOR_FIELD,  # The field containing document vectors
                "query_vector": query_vector,  # The encoded question vector
                "k": k,  # Number of final neighbors to return
                # Oversampling (num_candidates) improves accuracy at small speed cost
                "num_candidates": max(k * 10, 100),  
            },
            # Limit returned fields to save bandwidth
            "_source": ["PMID", "title", "content"],
            "size": k,
        }

        # 3. Execute the vector search
        response = self.es.search(index=DPR_INDEX_NAME, body=es_body)

        # 4. Process raw hits into our standard doc1, doc2... format
        results = {}
        for idx, hit in enumerate(response["hits"]["hits"], 1):
            results[f"doc{idx}"] = {
                "PMID": hit["_source"]["PMID"],
                "title": hit["_source"]["title"],
                "content": hit["_source"]["content"],
                "score": hit["_score"],  # The similarity score (higher is better)
            }

        # Serialize results to JSON
        return json.dumps(results, indent=4)


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    # Add project root to sys path to allow relative imports in direct run mode
    sys.path.insert(0, os.path.dirname(__file__))

    print("Initialising DPRRetriever (will download model on first run) …")
    retriever = DPRRetriever()

    query = "What are the structural proteins of a coronavirus?"
    print(f"Query: {query}\n")
    # Execute a sample retrieval
    result = retriever.retrieve_docs(query, k=3)
    data = json.loads(result)
    # Print formatted summary of retrieved docs
    for key, doc in data.items():
        print(f"{key}  PMID={doc['PMID']}  score={doc.get('score', '–'):.4f}")
        print(f"  Title  : {doc['title'][:80]}")
        print(f"  Content: {doc['content'][:120]}…\n")
