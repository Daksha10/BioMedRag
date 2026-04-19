"""
encode_documents_dpr.py
=======================
One-time preprocessing script that builds the semantic (vector) index for the RAG system.

Pipeline:
  1. Extraction: Reads every document from the standard keyword index (pubmed_index).
  2. Transformation: Concatenates title and content, then encodes them into 768-d vectors using DPR.
  3. Indexing: Creates a new Elasticsearch index (pubmed_dpr_index) with kNN vector support.
  4. Storage: Bulk-inserts the documents with their embeddings.
"""

import argparse  # Import argparse to handle command-line flags
import json  # Import JSON for structured logging and metadata
import os  # Import OS for path and environment access
import sys  # Import sys to modify the python search path
import time  # Import time for performance benchmarking

import numpy as np  # Import numpy for vector normalization and math
from dotenv import load_dotenv  # Import env for project-wide configuration
from elasticsearch import Elasticsearch, helpers  # Import core client and bulk helper
from tqdm import tqdm  # Import progress bar for terminal feedback

# ── PATH SETUP ───────────────────────────────────────────────────────────
# Calculate the project root automatically based on this file's depth
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Add 'rag_system' to the path so we can import our encoders (dpr_encoder.py)
sys.path.insert(0, os.path.join(REPO_ROOT, "rag_system"))

# Load global configuration from the .env file in the root
load_dotenv(os.path.join(REPO_ROOT, ".env"))

# ── SYSTEM CONSTANTS ──────────────────────────────────────────────────────
SOURCE_INDEX = os.getenv("INDEX_NAME", "pubmed_index")  # The keyword-based source index
DPR_INDEX = "pubmed_dpr_index"  # The new vector-based destination index
DPR_VECTOR_FIELD = "dpr_vector"  # The name of the field storing the embeddings
DPR_DIM = 768  # Dimensionality for DPR (fixed for this model)
SCROLL_SIZE = 500  # Number of docs to fetch per 'page' from Elasticsearch
SCROLL_TTL = "5m"  # Time-to-live for the Elasticsearch scroll cursor


# ── ELASTICSEARCH UTILITIES ──────────────────────────────────────────────

def get_es_client() -> Elasticsearch:
    """Creates a configured Elasticsearch client from .env variables."""
    es_host = os.getenv("ES_HOST", "http://localhost:9200")
    user = os.getenv("ELASTIC_USER", "elastic")
    password = os.getenv("ELASTIC_PASSWORD", "")
    cert_path = os.getenv("ES_CERT_PATH", "")

    # Handle SSL verification if a certificate is provided
    if cert_path and os.path.exists(cert_path):
        return Elasticsearch(
            es_host,
            basic_auth=(user, password),
            ca_certs=cert_path,
            verify_certs=True,
            request_timeout=120,
        )
    # Default connection for local development
    return Elasticsearch(
        es_host,
        basic_auth=(user, password),
        verify_certs=False,
        request_timeout=120,
    )


def create_dpr_index(es: Elasticsearch, recreate: bool = False):
    """Initializes the pubmed_dpr_index with specialized dense_vector mapping."""
    # Handle the --recreate flag to wipe existing data
    if es.indices.exists(index=DPR_INDEX):
        if recreate:
            print(f"Deleting existing '{DPR_INDEX}' index …")
            es.indices.delete(index=DPR_INDEX)
        else:
            print(f"Index '{DPR_INDEX}' already exists. Use --recreate to rebuild.")
            return False

    # Define the index schema (Mapping) for vector search
    mapping = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "PMID":    {"type": "keyword"},  # IDs should be keywords (not searchable by text)
                "title":   {"type": "text"},     # Standard text fields for visibility
                "content": {"type": "text"},
                DPR_VECTOR_FIELD: {
                    "type":       "dense_vector",  # Specialized vector type for ANN search
                    "dims":       DPR_DIM,         # 768 dimensions
                    "index":      True,            # Enable fast kNN indexing
                    "similarity": "cosine",        # Measure relevance via cosine distance
                },
            }
        },
    }
    # Send the creation request to Elasticsearch
    es.indices.create(index=DPR_INDEX, body=mapping)
    print(f"Created index '{DPR_INDEX}' with dense_vector({DPR_DIM}, cosine) mapping ✓")
    return True


def scroll_all_docs(es: Elasticsearch) -> list[dict]:
    """Efficiently streams all documents from the source index using Elasticsearch Scroll API."""
    all_docs = []
    # Initialize the scroll cursor
    page = es.search(
        index=SOURCE_INDEX,
        scroll=SCROLL_TTL,
        body={"size": SCROLL_SIZE, "_source": ["PMID", "title", "content"]},
    )
    scroll_id = page["_scroll_id"]

    while True:
        hits = page["hits"]["hits"]
        if not hits:
            break  # Exit when no more documents are found
        for hit in hits:
            src = hit["_source"]
            pmid = src.get("PMID") or ""
            all_docs.append({
                "es_id":   hit["_id"],  # Keep the internal Elasticsearch ID
                # Normalize PMID if it's missing or set to a literal string 'None'
                "PMID":    pmid if (pmid and pmid != "None") else hit["_id"],
                "title":   src.get("title", ""),
                "content": src.get("content", ""),
            })
        # Fetch the next page using the same scroll session
        page = es.scroll(scroll_id=scroll_id, scroll=SCROLL_TTL)
        scroll_id = page["_scroll_id"]

    # Close the scroll cursor on the server to free up resources
    try:
        es.clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass

    return all_docs


# ── MAIN EXECUTION LOGIC ──────────────────────────────────────────────

def main(args):
    """Main pipeline loop: Load docs -> Batch Encode -> Bulk Index."""
    es = get_es_client()

    # Safety check: Ensure the source data actually exists
    if not es.indices.exists(index=SOURCE_INDEX):
        print(f"ERROR: Source index '{SOURCE_INDEX}' not found in Elasticsearch.")
        print("  Run the ingestion script first:")
        print("  python information_retrieval/elastic_container/ingest_pubmed_subset.py")
        sys.exit(1)

    # Initialize the destination index
    created = create_dpr_index(es, recreate=args.recreate)
    if not created:
        count = es.count(index=DPR_INDEX)["count"]
        print(f"  '{DPR_INDEX}' contains {count:,} documents. Exiting.")
        sys.exit(0)

    # Load the heavy transformer model from disk or download from HuggingFace
    print("\nLoading DPR context encoder (first run downloads ~400 MB) …")
    from dpr_encoder import DPRPassageEncoder
    encoder = DPRPassageEncoder()
    print("Encoder loaded ✓\n")

    # Load all document text into memory for processing
    print(f"Fetching all documents from '{SOURCE_INDEX}' …")
    docs = scroll_all_docs(es)
    total = len(docs)
    print(f"Found {total:,} documents to encode.\n")

    # Initialize runtime trackers
    start = time.time()
    actions_buffer = [] # Accumulates documents ready for bulk insertion
    encoded_count = 0
    failed_count = 0

    def flush_buffer(buf):
        """Standard batch-sending function for Elasticsearch bulk indexing."""
        nonlocal encoded_count, failed_count
        ok = 0
        # helper streaming_bulk is highly performant for large datasets
        for success, info in helpers.streaming_bulk(
            es, buf, raise_on_error=False, chunk_size=len(buf)
        ):
            if success:
                ok += 1
            else:
                failed_count += 1
                if failed_count <= 3:
                    print(f"\n  ⚠️  Failed: {json.dumps(info)[:300]}")
        encoded_count += ok

    # Main iteration loop with Progress Bar
    with tqdm(total=total, unit="doc", desc="Encoding + indexing") as pbar:
        # Step through documents in target batch sizes (e.g., 16 at a time)
        for i in range(0, total, args.batch_size):
            batch = docs[i : i + args.batch_size]

            # Prepare text for the encoder (combining Title and Abstract)
            texts = [
                f"{d['title']} {d['content']}".strip() or "[empty]"
                for d in batch
            ]

            # BATCH ENCODING: The most time-consuming part (GPU/CPU bottleneck)
            vectors = encoder.encode_batch(texts, batch_size=args.batch_size)

            # NORMALIZATION: Scale vectors to unit length for Cosine Similarity
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vectors = (vectors / norms).astype("float32")

            # Map vectors back to their document objects in the bulk-action format
            for doc, vec in zip(batch, vectors):
                actions_buffer.append({
                    "_index": DPR_INDEX,
                    "_id":    doc["es_id"], 
                    "_source": {
                        "PMID":          doc["PMID"],
                        "title":         doc["title"],
                        "content":       doc["content"],
                        DPR_VECTOR_FIELD: vec.tolist(), # Convert numpy back to list/JSON
                    },
                })

            # Send to Elasticsearch once the buffer reaches es_batch size (e.g., 200)
            if len(actions_buffer) >= args.es_batch:
                flush_buffer(actions_buffer)
                actions_buffer = []

            pbar.update(len(batch))

    # Send any remaining documents in the final buffer
    if actions_buffer:
        flush_buffer(actions_buffer)

    # Force Elasticsearch to commit and refresh so data is searchable immediately
    es.indices.refresh(index=DPR_INDEX)

    # Print final execution metrics
    elapsed = time.time() - start
    final_count = es.count(index=DPR_INDEX)["count"]
    print(f"\n✅ Done! Indexed {encoded_count:,} | Failed {failed_count:,} | "
          f"Time {elapsed:.0f}s ({encoded_count/elapsed:.1f} docs/s)")
    print(f"   '{DPR_INDEX}' now contains {final_count:,} documents.")
    
    if failed_count > 0:
        print(f"   ⚠️  {failed_count} documents failed — re-run with --recreate to retry.")
    else:
        print("\nYou can now select 'DPR' in the Streamlit app or use DPRRetriever directly.")


# CLI ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the DPR dense-vector index in Elasticsearch."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Documents per GPU/CPU encoding batch (default: 16)",
    )
    parser.add_argument(
        "--es-batch",
        type=int,
        default=200,
        help="Documents per Elasticsearch bulk-insert batch (default: 200)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate pubmed_dpr_index if it already exists",
    )
    # Parse arguments and run main()
    main(parser.parse_args())
