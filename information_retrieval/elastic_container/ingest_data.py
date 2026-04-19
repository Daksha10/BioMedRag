#!/usr/bin/env python

import json
import os
from pathlib import Path
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv

load_dotenv()

password = os.getenv("ELASTIC_PASSWORD")
user = os.getenv("ELASTIC_USER", "elastic")
es_host = os.getenv("ES_HOST", "https://localhost:9200")
cert_path = os.getenv("ES_CERT_PATH")

if cert_path and os.path.exists(cert_path):
    es = Elasticsearch(
        hosts=[es_host],
        ca_certs=cert_path,
        basic_auth=(user, password),
        verify_certs=True
    )
else:
    es = Elasticsearch(
        hosts=[es_host],
        basic_auth=(user, password),
        verify_certs=False
    )

# Define the index name
index_name = "pubmed_index"

# Delete the index if it exists
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# Check again if the index exists, and if not, create it
if not es.indices.exists(index=index_name):
    # Define the mapping
    mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "default": {
                    "type": "standard",  
                    "stopwords": "_english_" 
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "default",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256 
                    }
                }
            }
        }
    }
}


# Create the index with the defined mapping
es.indices.create(index=index_name, body=mapping)

source_directory = Path(os.getenv('SOURCE_DATA_DIR', './sample_data'))
error_log_path = Path('./errors.jsonl')  # Path to error log file

def bulk_index_documents(source_directory, index_name, error_log_path):
    if not source_directory.exists():
        print(f"Error: The source directory '{source_directory}' does not exist.")
        return

    print(f"Ingesting documents from '{source_directory.absolute()}'...")
    actions = []  # List to store the documents to be indexed
    file_count = 0
    doc_count = 0

    # Open the error log file for writing
    with error_log_path.open('w') as error_log:
        # Iterate through each file in the source directory
        files = list(os.listdir(source_directory))
        print(f"Found {len(files)} files in directory.")
        
        for file_name in tqdm(files):
            if file_name.endswith('.jsonl'):
                file_count += 1
                source_file = source_directory / file_name
                
                # Open and read the JSONL file
                with open(source_file, 'r') as json_file:
                    for line_num, line in enumerate(json_file, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            doc = json.loads(line)
                            
                            action = {
                                "_index": index_name,
                                "_source": doc
                            }
                            actions.append(action)
                            doc_count += 1

                            if len(actions) == 200:  # Bulk indexing threshold
                                helpers.bulk(es, actions)
                                actions = []
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON in {file_name} at line {line_num}")
                            error_log.write(f"Error in file {file_name}: {e}\n")
                            error_log.write(f"{line}\n")
                        except Exception as e:
                            print(f"Unexpected error in {file_name} at line {line_num}: {e}")
                            error_log.write(f"Unexpected error in file {file_name}: {e}\n")
                            error_log.write(f"{line}\n")

        # Index any remaining documents
        if actions:
            helpers.bulk(es, actions)

    print(f'Indexing complete. Processed {file_count} files and {doc_count} documents.')

# Call the function to index the documents
bulk_index_documents(source_directory, index_name, error_log_path)

# Refresh the index to make sure documents are searchable before counting
es.indices.refresh(index=index_name)

# Count and print the number of documents in the index
count_result = es.count(index=index_name)
print(f"Index contains {count_result['count']} documents.")

# to run this script in the background, use the following command:
# nohup python3 ./ingest_data.py > output.log 2>&1 &