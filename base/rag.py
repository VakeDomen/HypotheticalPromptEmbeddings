import json
import faiss
import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

print("Loading chunked data...")
with open('../data/chunked_data.json', 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)

print("Loading models...")
embedding_model = OllamaEmbeddings(base_url="hivecore.famnit.upr.si:6666", model='bge-m3')
llm = OllamaLLM(base_url="hivecore.famnit.upr.si:6666", model='mistral-nemo')

print("Preparing for indexing...")
texts = []
metadata = []
for doc_id, item in tqdm(enumerate(chunked_data), total=len(chunked_data)):
    for chunk_id, chunk in enumerate(item['chunks']):
        texts.append(chunk)
        metadata.append({
            'doc_id': item['context_id'],
            'chunk_id': chunk_id,
            'Q': item['Q'],
            'A': item['A']
        })

print("Embedding chunks...")

def embed_text(i, text):
    embedding = embedding_model.embed_documents([text])
    return i, embedding[0]  # Since embed_documents returns a list

embeddings = [None] * len(texts)  # Pre-allocate a list for embeddings

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(embed_text, i, text): i for i, text in enumerate(texts)}
    
    for future in tqdm(as_completed(futures), total=len(futures)):
        i, embedding = future.result()
        embeddings[i] = embedding

embeddings = np.array(embeddings).astype('float32')


print("Creating index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def retrieve_documents(question, top_k=5):
    question_embedding = embedding_model.embed_documents([question])
    question_embedding = np.array(question_embedding).astype('float32')
    D, I = index.search(question_embedding, top_k)
    retrieved_docs = [texts[i] for i in I[0]]
    retrieved_metadata = [metadata[i] for i in I[0]]
    return retrieved_docs, retrieved_metadata

def generate_final_answer(question, context):
    context_text = "\n".join(context)
    prompt = f"Answer the following question using the provided context.\n\nContext:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer:"
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