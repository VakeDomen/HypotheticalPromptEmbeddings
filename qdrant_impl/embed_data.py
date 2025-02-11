import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings
import numpy as np
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

BATCH_SIZE = 100
MAX_RETRIES = 10

print("Loading data from 'chunks_with_questions.json'...")
with open('chunks_with_questions.json', 'r', encoding='utf-8') as f:
    chunks_questions = json.load(f)

print("Connecting to Qdrant...")
client = QdrantClient(host="localhost", port=6333)

print("Loading embedding model...")
embedding_model = OllamaEmbeddings(model='bge-m3')

COLLECTION_PREFIX = "multihop_"
distance_metrics = ['euclidean', "cosine"]
distance_mapping = {
    'cosine': qdrant_models.Distance.COSINE,
    'euclidean': qdrant_models.Distance.EUCLID
}
dimension = 1024

def create_collections_and_disable_indexing():
    """Create or recreate the collections and disable indexing by setting a very high full_scan_threshold."""
    for distance_metric in distance_metrics:
        collection_name_chunks = f"{COLLECTION_PREFIX}chunks_{distance_metric}"
        print(f"Creating Qdrant collection '{collection_name_chunks}'...")
        client.recreate_collection(
            collection_name=collection_name_chunks,
            vectors_config=qdrant_models.VectorParams(
                size=dimension,
                distance=distance_mapping[distance_metric]
            ),
        )
        
        # Disable indexing by setting a high full_scan_threshold
        client.update_collection(
            collection_name=collection_name_chunks,
            hnsw_config=qdrant_models.HnswConfigDiff(full_scan_threshold=1_000_000_000)  # large number
        )

        collection_name_questions = f"{COLLECTION_PREFIX}questions_{distance_metric}"
        print(f"Creating Qdrant collection '{collection_name_questions}'...")
        client.recreate_collection(
            collection_name=collection_name_questions,
            vectors_config=qdrant_models.VectorParams(
                size=dimension,
                distance=distance_mapping[distance_metric]
            ),
        )

        # Disable indexing for questions as well
        client.update_collection(
            collection_name=collection_name_questions,
            hnsw_config=qdrant_models.HnswConfigDiff(full_scan_threshold=1_000_000_000)
        )

def reenable_indexing():
    """Re-enable indexing by setting full_scan_threshold to a normal value after bulk inserts."""
    for distance_metric in distance_metrics:
        collection_name_chunks = f"{COLLECTION_PREFIX}chunks_{distance_metric}"
        collection_name_questions = f"{COLLECTION_PREFIX}questions_{distance_metric}"

        # Re-enable indexing by lowering the threshold
        client.update_collection(
            collection_name=collection_name_chunks,
            hnsw_config=qdrant_models.HnswConfigDiff(full_scan_threshold=10_000)
        )
        client.update_collection(
            collection_name=collection_name_questions,
            hnsw_config=qdrant_models.HnswConfigDiff(full_scan_threshold=10_000)
        )


create_collections_and_disable_indexing()

def process_chunk(chunk_info):
    doc_id = chunk_info['doc_id']
    chunk_id = chunk_info['chunk_id']
    chunk_text = chunk_info['chunk_text']
    generated_questions = chunk_info.get('generated_questions', [])

    texts_to_embed = [chunk_text] + generated_questions
    for attempt in range(MAX_RETRIES):
        try:
            embeddings = embedding_model.embed_documents(texts_to_embed)
            break
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Failed to embed chunk {doc_id}_c{chunk_id} after {MAX_RETRIES} attempts: {e}")
                return None

    chunk_embedding = embeddings[0]
    question_embeddings = embeddings[1:]

    # Metadata for the chunk
    point_uuid_chunk = str(uuid.uuid4())
    metadata_chunk = {
        'doc_id': doc_id,
        'chunk_id': chunk_id,
        'text': chunk_text
    }

    # Prepare question embeddings and metadata
    question_ids = []
    question_vectors = []
    question_payloads = []
    for index, (question, question_embedding) in enumerate(zip(generated_questions, question_embeddings)):
        point_uuid_question = str(uuid.uuid4())
        metadata_question = {
            'doc_id': doc_id,
            'chunk_id': chunk_id,
            'text': chunk_text,
            'question_index': index,
            'generated_question': question,
        }
        question_ids.append(point_uuid_question)
        question_vectors.append(question_embedding)
        question_payloads.append(metadata_question)

    return {
        "chunk_id": point_uuid_chunk,
        "chunk_vector": chunk_embedding,
        "chunk_payload": metadata_chunk,
        "question_ids": question_ids,
        "question_vectors": question_vectors,
        "question_payloads": question_payloads
    }

print("Embedding data in parallel...")
results = []
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(process_chunk, chunk_info) for chunk_info in chunks_questions]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding chunks"):
        result = future.result()
        if result is not None:
            results.append(result)
        else:
            print("A chunk failed to process and was skipped.")

print("Preparing batched uploads...")

# Separate data by metric for chunks and questions
chunks_data = {dm: {"ids": [], "vectors": [], "payloads": []} for dm in distance_metrics}
questions_data = {dm: {"ids": [], "vectors": [], "payloads": []} for dm in distance_metrics}

for res in results:
    # Add chunk data to each metric collection
    for dm in distance_metrics:
        chunks_data[dm]["ids"].append(res["chunk_id"])
        chunks_data[dm]["vectors"].append(res["chunk_vector"])
        chunks_data[dm]["payloads"].append(res["chunk_payload"])

    # Add question data to each metric collection
    for dm in distance_metrics:
        questions_data[dm]["ids"].extend(res["question_ids"])
        questions_data[dm]["vectors"].extend(res["question_vectors"])
        questions_data[dm]["payloads"].extend(res["question_payloads"])

print(f"Uploading all chunks and questions in batches of size {BATCH_SIZE} to Qdrant...")

def upload_in_batches(client, collection_name, ids, vectors, payloads, batch_size, max_retries):
    total_points = len(ids)
    if total_points == 0:
        return

    with tqdm(total=total_points, desc=f"Uploading to {collection_name}", unit="points") as pbar:
        for i in range(0, total_points, batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_vectors = vectors[i:i+batch_size]
            batch_payloads = payloads[i:i+batch_size]

            # Retry logic for uploading the batch
            for attempt in range(max_retries):
                try:
                    client.upsert(
                        collection_name=collection_name,
                        points=qdrant_models.Batch(
                            ids=batch_ids,
                            vectors=batch_vectors,
                            payloads=batch_payloads
                        )
                    )
                    break  # Success
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to batch upload to '{collection_name}' after {max_retries} attempts: {e}")
                    else:
                        print(f"Error uploading to '{collection_name}': {e}. Retrying ({attempt+1}/{max_retries})...")
            pbar.update(len(batch_ids))

for distance_metric in distance_metrics:
    collection_name_chunks = f"{COLLECTION_PREFIX}chunks_{distance_metric}"
    collection_name_questions = f"{COLLECTION_PREFIX}questions_{distance_metric}"

    # Upsert chunks in batches
    upload_in_batches(
        client,
        collection_name_chunks,
        chunks_data[distance_metric]["ids"],
        chunks_data[distance_metric]["vectors"],
        chunks_data[distance_metric]["payloads"],
        BATCH_SIZE,
        MAX_RETRIES
    )

    # Upsert questions in batches
    upload_in_batches(
        client,
        collection_name_questions,
        questions_data[distance_metric]["ids"],
        questions_data[distance_metric]["vectors"],
        questions_data[distance_metric]["payloads"],
        BATCH_SIZE,
        MAX_RETRIES
    )

print("Re-enabling indexing on collections...")
reenable_indexing()

print("All embeddings and batch uploads completed. Indexing has been re-enabled.")
