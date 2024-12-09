import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
import numpy as np
import faiss

# Step 1: Load 'chunks_with_questions.json' instead of regenerating
print("Loading chunks with generated questions...")
with open('chunks_with_questions.json', 'r', encoding='utf-8') as f:
    chunks_questions = json.load(f)

print("Loading models...")
embedding_model = OllamaEmbeddings(base_url="hivecore.famnit.upr.si:6666", model='bge-m3')
llm = OllamaLLM(base_url="hivecore.famnit.upr.si:6666", model='mistral-nemo')

# Step 2: Load 'chunked_data.json' to retrieve 'Q' and 'A' for metadata
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

# Step 3: Reconstruct 'texts', 'metadata', and 'questions_for_embedding' from loaded data
print("Reconstructing texts, metadata, and questions_for_embedding...")
texts = []
metadata = []
questions_for_embedding = []

for chunk_info in chunks_questions:
    doc_id = chunk_info['doc_id']
    chunk_id = chunk_info['chunk_id']
    chunk_text = chunk_info['chunk_text']
    generated_questions = chunk_info['generated_questions']
    QA = context_id_to_QA.get(doc_id, {'Q': '', 'A': ''})
    for index, question in enumerate(generated_questions):
        texts.append(chunk_text)
        metadata.append({
            'doc_id': f"x{doc_id}_c{chunk_id}_q{str(index)}",
            'chunk_id': chunk_id,
            'Q': QA['Q'],
            'A': QA['A']
        })
        questions_for_embedding.append(question)

# Proceed with embedding and indexing
print("Embedding chunks...")

def embed_text_batch(batch):
    indices, texts_batch = zip(*batch)
    try:
        embeddings_batch = embedding_model.embed_documents(texts_batch)
        return list(zip(indices, embeddings_batch))
    except Exception as e:
        print(f"An error occurred while embedding batch starting at index {indices[0]}: {e}")
        return None

# Adjusted batch size to prevent overloading the server
batch_size = 5

# Prepare batches for embedding chunks (texts)
texts_batches = [
    [(i, texts[i]) for i in range(j, min(j + batch_size, len(texts)))]
    for j in range(0, len(texts), batch_size)
]

embeddings = [None] * len(texts)  # Pre-allocate a list for embeddings

print("Embedding in parallel...")
with ThreadPoolExecutor(max_workers=50) as executor:  # Limit max_workers to prevent overload
    futures = [executor.submit(embed_text_batch, batch) for batch in texts_batches]
    for future in tqdm(as_completed(futures), total=len(futures)):
        batch_results = future.result()
        if batch_results:
            for i, embedding in batch_results:
                embeddings[i] = embedding
        else:
            print("A batch failed to process and was skipped.")

# Remove any None values that may have resulted from failed batches
embeddings = [emb for emb in embeddings if emb is not None]
embeddings = np.array(embeddings).astype('float32')

print("Creating index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Functions for retrieval and answer generation
def retrieve_documents(question, top_k=5):
    question_embedding = embedding_model.embed_documents([question])
    question_embedding = np.array(question_embedding).astype('float32')
    D, I = index.search(question_embedding, top_k)
    retrieved_docs = [texts[i] for i in I[0]]  # Retrieve the original chunk
    retrieved_metadata = [metadata[i] for i in I[0]]
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

max_workers = 50  # Reduce the number of workers to prevent freezing
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
