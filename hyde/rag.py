import json
import faiss
import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

print("Loading chunked data...")
data = '../data/chunked_data.json'
if len(sys.argv) > 1:
    data = sys.argv[1]
with open(data, 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)

print("Loading models...")
embedding_model = OllamaEmbeddings(base_url="hivecore.famnit.upr.si:6666", model='bge-m3')
llm = OllamaLLM(base_url="hivecore.famnit.upr.si:6666", model='mistral-nemo')

print("Preparing for indexing...")
texts, metadata = zip(*[
    (chunk, {
        'doc_id': item['context_id'],
        'chunk_id': chunk_id,
        'Q': item['Q'],
        'A': item['A']
    })
    for item in tqdm(chunked_data, total=len(chunked_data))
    for chunk_id, chunk in enumerate(item['chunks'])
])

# Convert tuples back to lists if necessary
texts = list(texts)
metadata = list(metadata)

# Set batch size
batch_size = 5

def embed_text_batch(batch):
    indices, texts_batch = zip(*batch)
    embeddings_batch = embedding_model.embed_documents(texts_batch)
    return list(zip(indices, embeddings_batch))

# Create batches of indices and texts
texts_batches = [
    [(i, texts[i]) for i in range(j, min(j + batch_size, len(texts)))]
    for j in range(0, len(texts), batch_size)
]

embeddings = [None] * len(texts)  # Pre-allocate a list for embeddings

print("Embedding chunks in batches...")
with ThreadPoolExecutor(max_workers=50) as executor:
    futures = {executor.submit(embed_text_batch, batch): batch for batch in texts_batches}
    
    for future in tqdm(as_completed(futures), total=len(futures)):
        batch_results = future.result()
        for i, embedding in batch_results:
            embeddings[i] = embedding

embeddings = np.array(embeddings).astype('float32')



print("Creating index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def generate_hypothetical_answer(question):
    prompt = f"Please write a passage to answer the question.\nQuestion: {question}\nPassage:"
    hypothetical_answer = llm.invoke(prompt)
    return hypothetical_answer

def retrieve_documents(question, top_k=5):
    hypo_answer = generate_hypothetical_answer(question)
    hypo_embedding = embedding_model.embed_documents([hypo_answer])
    hypo_embedding = np.array(hypo_embedding).astype('float32')
    D, I = index.search(hypo_embedding, top_k)
    retrieved_docs = [texts[i] for i in I[0]]
    retrieved_metadata = [metadata[i] for i in I[0]]
    return retrieved_docs, retrieved_metadata

def generate_final_answer(question, context):
    context_text = "\n".join(context)
    prompt = f"Answer the following question using the provided context. If no answer can be found in the context, answer 'No answer avalible'.\n\nContext:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer:"
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

print("Answering questions using HyDE RAG with multithreading...")

max_workers = 30  
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
