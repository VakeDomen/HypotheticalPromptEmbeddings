import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
import numpy as np

# Import Qdrant Client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# Import uuid module
import uuid

# Step 1: Load 'chunks_with_questions.json' instead of regenerating
print("Loading chunks with generated questions...")
with open('chunks_with_questions.json', 'r', encoding='utf-8') as f:
    chunks_questions = json.load(f)
print("Loading models...")
embedding_model = OllamaEmbeddings(base_url="hivecore.famnit.upr.si:6666", model='bge-m3')
llm = OllamaLLM(base_url="hivecore.famnit.upr.si:6666", model='mistral-nemo')

# Step 2: Load 'chunked_data.json' to retrieve Q and A for metadata
print("Loading chunked data to retrieve Q and A for metadata...")
data = '../data/chunked_data.json'
if len(sys.argv) > 1:
    data = sys.argv[1]
with open(data, 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)

# Build a mapping from 'context_id' to 'Q' and 'A'
context_id_to_QA = {}
for item in chunked_data:
    context_id = item['context_id']
    Q = item.get('Q', '')
    A = item.get('A', '')
    context_id_to_QA[context_id] = {'Q': Q, 'A': A}

# Step 3: Reconstruct data for embedding...
print("Reconstructing data for embedding...")
data_for_embedding = []

for chunk_info in chunks_questions:
    doc_id = chunk_info['doc_id']
    chunk_id = chunk_info['chunk_id']
    chunk_text = chunk_info['chunk_text']
    generated_questions = chunk_info['generated_questions']
    QA = context_id_to_QA.get(doc_id, {'Q': '', 'A': ''})
    for index, question in enumerate(generated_questions):
        point_uuid = str(uuid.uuid4())  # Generate a unique UUID for each point
        metadata = {
            'doc_id': f"x{doc_id}_c{chunk_id}_q{str(index)}",
            'chunk_id': chunk_id,
            'Q': QA['Q'],
            'A': QA['A'],
            'text': chunk_text,  # Include text in the payload
            'generated_question': question,
        }
        data_for_embedding.append((point_uuid, question, metadata))

# Proceed with embedding and uploading to Qdrant...
print("Connecting to Qdrant...")
client = QdrantClient(host="localhost", port=6333)  # Adjust host and port if necessary

collection_name = "ms_marco_questions"
dimension = 1024  # Or set dimension explicitly if known

print("Creating Qdrant collection...")
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=qdrant_models.VectorParams(size=dimension, distance=qdrant_models.Distance.COSINE),
)

print("Embedding and uploading in parallel...")

def embed_and_upload_batch(batch):
    ids, texts_batch, metadata_batch = zip(*batch)
    try:
        embeddings_batch = embedding_model.embed_documents(texts_batch)
        # Upload embeddings to Qdrant
        payloads = list(metadata_batch)
        client.upsert(
            collection_name=collection_name,
            points=qdrant_models.Batch(
                ids=ids,  # Pass IDs as UUID strings
                vectors=embeddings_batch,
                payloads=payloads
            )
        )
        return True
    except Exception as e:
        print(f"An error occurred while embedding and uploading batch starting with ID {ids[0]}: {e}")
        return False

# Adjusted batch size to prevent overloading the server
batch_size = 5

# Prepare batches for embedding and uploading
data_batches = [
    data_for_embedding[i:i + batch_size]
    for i in range(0, len(data_for_embedding), batch_size)
]

with ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(embed_and_upload_batch, batch) for batch in data_batches]
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        if not result:
            print("A batch failed to process and was skipped.")

# Functions for retrieval and answer generation
def retrieve_documents(question, top_k=5):
    question_embedding = embedding_model.embed_documents([question])[0]  # Get the embedding vector
    # Perform search in Qdrant
    search_result = client.search(
        collection_name=collection_name,
        query_vector=question_embedding,
        limit=top_k,
    )
    # Extract the retrieved documents and metadata
    retrieved_docs = []
    retrieved_metadata = []
    for hit in search_result:
        payload = hit.payload
        retrieved_docs.append(payload['text'])
        retrieved_metadata.append(payload)
    return retrieved_docs, retrieved_metadata

def generate_final_answer(question, context):
    context_text = "\n".join(context)
    prompt = f"Answer the following question using the provided context. If no answer can be found in the context, answer 'No answer available'.\n\nContext:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer:"
    final_answer = llm.invoke(prompt)
    return final_answer

def process_question(query_id, item):
    question = item['Q']
    gt_answer = item['A']
    if not question or not gt_answer:
        return None

    retrieved_docs, retrieved_metadata = retrieve_documents(question)
    response = generate_final_answer(question, retrieved_docs)

    retrieved_context = []
    for md, doc in zip(retrieved_metadata, retrieved_docs):
        retrieved_context.append({
            'doc_id': f"{md['doc_id']}_{md['chunk_id']}",
            'text': doc
        })

    result = {
        'query_id': f"{query_id:03d}",
        'query': question,
        'gt_answer': gt_answer,
        'response': response.strip(),
        'retrieved_context': retrieved_context
    }
    return result

results = []

print("Answering questions using Base RAG with multithreading...")

max_workers = 50  # Adjust based on your system capabilities
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_query_id = {
        executor.submit(process_question, query_id, item): query_id
        for query_id, item in enumerate(chunked_data)
    }

    for future in tqdm(as_completed(future_to_query_id), total=len(future_to_query_id)):
        try:
            result = future.result()
            if result:
                results.append(result)
        except Exception as e:
            query_id = future_to_query_id[future]
            print(f"An error occurred while processing query_id {query_id}: {e}")

print("Saving results...")
output_data = {'results': results}
with open('rag_results.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("Done!")
