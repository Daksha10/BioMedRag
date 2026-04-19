import os
import json
import time
from datasets import load_dataset
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment
ES_HOST = os.getenv('ES_HOST', 'http://localhost:9200')
ELASTIC_PASSWORD = os.getenv('ELASTIC_PASSWORD', 'elastic_admin_password')
ELASTIC_USER = os.getenv('ELASTIC_USER', 'elastic')
INDEX_NAME = os.getenv('INDEX_NAME', 'pubmed_index')
ES_CERT_PATH = os.getenv('ES_CERT_PATH', '')

def get_es_client():
    if ES_CERT_PATH:
        return Elasticsearch(
            ES_HOST,
            basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
            ca_certs=ES_CERT_PATH,
            verify_certs=True
        )
    else:
        return Elasticsearch(
            ES_HOST,
            basic_auth=(ELASTIC_USER, ELASTIC_PASSWORD),
            verify_certs=False
        )

def ingest_subset(limit=10000):
    es = get_es_client()
    
    print(f"Connecting to Hugging Face dataset 'slinusc/PubMedAbstractsSubset'...")
    # Stream the dataset to avoid large downloads
    dataset = load_dataset('slinusc/PubMedAbstractsSubset', split='train', streaming=True)
    
    # Ensure index exists
    if not es.indices.exists(index=INDEX_NAME):
        print(f"Creating index '{INDEX_NAME}'...")
        es.indices.create(index=INDEX_NAME)

    actions = []
    count = 0
    
    print(f"Ingesting up to {limit} documents into '{INDEX_NAME}'...")
    
    start_time = time.time()
    for record in tqdm(dataset, total=limit):
        if count >= limit:
            break
            
        # Standardize record structure for the RAG system
        doc = {
            "_index": INDEX_NAME,
            "_id": record.get('pmid'),
            "_source": {
                "PMID": str(record.get('pmid')),
                "title": record.get('title', ''),
                "content": record.get('abstract', record.get('content', '')),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{record.get('pmid')}/"
            }
        }
        actions.append(doc)
        count += 1
        
        # Bulk ingest in batches of 500
        if len(actions) >= 500:
            helpers.bulk(es, actions)
            actions = []

    # Final batch
    if actions:
        helpers.bulk(es, actions)
    
    # Refresh to make searchable
    es.indices.refresh(index=INDEX_NAME)
    
    end_time = time.time()
    print(f"\nSuccessfully ingested {count} documents.")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")
    
    # Verify count
    res = es.count(index=INDEX_NAME)
    print(f"Total documents now in index: {res['count']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest a subset of PubMed abstracts from Hugging Face.")
    parser.add_argument("--limit", type=int, default=1000, help="Number of documents to ingest (default: 1000)")
    args = parser.parse_args()
    
    ingest_subset(limit=args.limit)
