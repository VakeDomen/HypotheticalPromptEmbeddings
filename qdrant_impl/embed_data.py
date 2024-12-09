import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings
import numpy as np
import uuid

# Import Qdrant Client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# No input parameters as per your instructions

print("Loading data from 'chunks_with_questions.json'...")
with open('chunks_with_questions.json', 'r', encoding='utf-8') as f:
    chunks_questions = json.load(f)

print("Connecting to Qdrant...")
client = QdrantClient(host="localhost", port=6333) 


print("Loading embedding model...")
embedding_model = OllamaEmbeddings(base_url="hivecore.famnit.upr.si:6666", model='bge-m3')


COLLECTION_PREFIX="rag_10k_"
# Define distance metrics and their mapping
distance_metrics = ['cosine', 'euclidean']
distance_mapping = {
    'cosine': qdrant_models.Distance.COSINE,
    'euclidean': qdrant_models.Distance.EUCLID
}

dimension = 1024


# Create or recreate the collections
for distance_metric in distance_metrics:
    collection_name_chunks = f"{COLLECTION_PREFIX}chunks_{distance_metric}"
    print(f"Creating Qdrant collection '{collection_name_chunks}' with distance metric '{distance_metric}'...")
    client.recreate_collection(
        collection_name=collection_name_chunks,
        vectors_config=qdrant_models.VectorParams(size=dimension, distance=distance_mapping[distance_metric]),
    )

    collection_name_questions = f"{COLLECTION_PREFIX}questions_{distance_metric}"
    print(f"Creating Qdrant collection '{collection_name_questions}' with distance metric '{distance_metric}'...")
    client.recreate_collection(
        collection_name=collection_name_questions,
        vectors_config=qdrant_models.VectorParams(size=dimension, distance=distance_mapping[distance_metric]),
    )


def process_chunk(chunk_info):
    doc_id = chunk_info['doc_id']
    chunk_id = chunk_info['chunk_id']
    chunk_text = chunk_info['chunk_text']
    generated_questions = chunk_info.get('generated_questions', [])

    # Prepare all texts to embed: the chunk text and the generated questions
    texts_to_embed = [chunk_text] + generated_questions 

    max_retries = 5
    for attempt in range(max_retries):
        try:
            embeddings = embedding_model.embed_documents(texts_to_embed)
            break 
        except Exception as e:
            print(f"An error occurred while embedding chunk {doc_id}_c{chunk_id} and its questions: {e}")
            print(f"Retrying embedding ({attempt + 1}/{max_retries})...")
            if attempt == max_retries - 1:
                print(f"Failed to embed chunk {doc_id}_c{chunk_id} and its questions after {max_retries} attempts.")
                return False
    else:
        return False

    # Extract the chunk embedding and question embeddings
    chunk_embedding = embeddings[0]
    question_embeddings = embeddings[1:]

    # Generate a unique ID for the chunk
    point_uuid_chunk = str(uuid.uuid4())

    # Metadata for the chunk
    metadata_chunk = {
        'doc_id': doc_id,
        'chunk_id': chunk_id,
        'text': chunk_text
    }

    # Prepare the question embeddings and metadata
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

    # upload the chunk embedding to both 'chunks' collections
    for distance_metric in distance_metrics:
        collection_name_chunks = f"{COLLECTION_PREFIX}chunks_{distance_metric}"
        # Retry logic for uploading to Qdrant
        for attempt in range(max_retries):
            try:
                client.upsert(
                    collection_name=collection_name_chunks,
                    points=qdrant_models.Batch(
                        ids=[point_uuid_chunk],
                        vectors=[chunk_embedding],
                        payloads=[metadata_chunk]
                    )
                )
                break  
            except Exception as e:
                print(f"An error occurred while uploading chunk {metadata_chunk['doc_id']} to collection '{collection_name_chunks}': {e}")
                print(f"Retrying chunk upload ({attempt + 1}/{max_retries})...")
                if attempt == max_retries - 1:
                    print(f"Failed to upload chunk {metadata_chunk['doc_id']} to collection '{collection_name_chunks}' after {max_retries} attempts.")

    # Now, upload the question embeddings to both 'questions' collections
    for distance_metric in distance_metrics:
        collection_name_questions = f"{COLLECTION_PREFIX}questions_{distance_metric}"
        # Retry logic for uploading to Qdrant
        for attempt in range(max_retries):
            try:
                client.upsert(
                    collection_name=collection_name_questions,
                    points=qdrant_models.Batch(
                        ids=question_ids,
                        vectors=question_vectors,
                        payloads=question_payloads
                    )
                )
                break  # Exit the loop if upload is successful
            except Exception as e:
                print(f"An error occurred while uploading questions for chunk {doc_id}_c{chunk_id} to collection '{collection_name_questions}': {e}")
                print(f"Retrying question batch upload ({attempt + 1}/{max_retries})...")
                if attempt == max_retries - 1:
                    print(f"Failed to upload questions for chunk {doc_id}_c{chunk_id} to collection '{collection_name_questions}' after {max_retries} attempts.")
        

    return True

print("Embedding and uploading data in parallel...")

with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(process_chunk, chunk_info) for chunk_info in chunks_questions]
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        if not result:
            print("A chunk failed to process and was skipped.")

print("All embeddings and uploads completed.")
