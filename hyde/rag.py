import json
import faiss
import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

print("Loading chunked data...")
with open('../data/chunked_data.json', 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)


chunked_data = chunked_data[:10]

print("Loading models...")
embedding_model = OllamaEmbeddings(base_url="prog3.student.famnit.upr.si:6666", model='bge-m3')
llm = OllamaLLM(base_url="prog3.student.famnit.upr.si:6666", model='mistral-nemo')


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
embeddings = embedding_model.embed_documents(texts)
embeddings = np.array(embeddings).astype('float32')

print("Creating index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


def generate_hypothetical_answer(question):
    prompt = f"Question: {question}\nAnswer:"
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
    prompt = f"Answer the following question using the provided context.\n\nContext:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer:"
    final_answer = llm.invoke(prompt)
    return final_answer

results = []

print("Answering questions using HyDE RAG...")
for query_id, item in tqdm(enumerate(chunked_data), total=len(chunked_data)):
    question = item['Q']
    gt_answer = item['A']
    if not question or not gt_answer:
        continue  # Skip if question or answer is missing

    # Retrieve documents using HyDE
    retrieved_docs, retrieved_metadata = retrieve_documents(question)

    # Generate final answer
    response = generate_final_answer(question, retrieved_docs)

    # Prepare retrieved context with doc_ids
    retrieved_context = []
    for md, doc in zip(retrieved_metadata, retrieved_docs):
        retrieved_context.append({
            'doc_id': f"{md['doc_id']}_{md['chunk_id']}",
            'text': doc
        })

    # Append to results
    results.append({
        'query_id': f"{query_id:03d}",
        'query': question,
        'gt_answer': gt_answer,
        'response': response.strip(),
        'retrieved_context': retrieved_context
    })

# Save results to JSON
output_data = {'results': results}
with open('rag_results.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
