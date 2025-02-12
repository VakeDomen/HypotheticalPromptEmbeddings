import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

# Import Qdrant Client
from qdrant_client import QdrantClient

# Collections mapping for chunks (same as base retrieval)
COLLECTION_PREFIX="ragbench_"
K=10
collections = {
    'cosine': f"{COLLECTION_PREFIX}chunks_cosine",
    #'euclidean': f"{COLLECTION_PREFIX}chunks_euclidean",
}

print("Loading chunked data to retrieve Q and A for answering...")
data = '../data/chunked_data.json'
if len(sys.argv) > 1:
    data = sys.argv[1]
with open(data, 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)


print("Loading models...")
embedding_model = OllamaEmbeddings(model='bge-m3')
llm = OllamaLLM(model='mistral-nemo')

print("Connecting to Qdrant...")
client = QdrantClient(host="localhost", port=6333)

def generate_hypothetical_answer(question):
    max_retries = 10
    prompt = f"Please write a passage to answer the question.\nQuestion: {question}\nPassage:"
    for attempt in range(max_retries):
        try:
            hypothetical_answer = llm.invoke(prompt)
            break  # Exit loop if successful
        except Exception as e:
            time.sleep(1)  # Optional delay
            if attempt == max_retries - 1:
                print(f"An error occurred while generating hypothetical answer: {e}")
                print(f"Retrying hypothetical answer generation ({attempt + 1}/{max_retries})...")
                print(f"Failed to generate hypothetical answer after {max_retries} attempts.")
                return None
    else:
        return None
    return hypothetical_answer.strip()

def retrieve_documents(hypothetical_answer_embedding, collection_name, top_k=K):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Search in Qdrant
            search_result = client.search(
                collection_name=collection_name,
                query_vector=hypothetical_answer_embedding,
                limit=top_k,
            )
            break  # Exit loop if successful
        except Exception as e:
            time.sleep(1)  # Optional delay
            if attempt == max_retries - 1:
                print(f"An error occurred during Qdrant search: {e}")
                print(f"Retrying search ({attempt + 1}/{max_retries})...")
                print(f"Failed to search Qdrant after {max_retries} attempts.")
                return [], []

    retrieved_docs = []
    retrieved_metadata = []
    for hit in search_result:
        payload = hit.payload
        retrieved_docs.append(payload['text'])
        retrieved_metadata.append(payload)
    return retrieved_docs, retrieved_metadata

def generate_final_answer(question, context):
    max_retries = 10
    context_text = "\n".join(context)
    prompt = f"Answer the following question using the provided context. \n\nContext:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer:"
    for attempt in range(max_retries):
        try:
            final_answer = llm.invoke(prompt)
            break  # Exit loop if successful
        except Exception as e:
            
            time.sleep(1)  # Optional delay
            if attempt == max_retries - 1:
                print(f"An error occurred while invoking the LLM: {e}")
                print(f"Retrying LLM invocation ({attempt + 1}/{max_retries})...")
                print(f"Failed to invoke LLM after {max_retries} attempts.")
                return "No answer available"
    else:
        return "No answer available"

    return final_answer.strip()

def process_question(query_id, item):
    question = item['Q']
    gt_answer = item['A']
    if not question or not gt_answer:
        return None

    results_per_collection = {}

    # Generate hypothetical answer
    hypothetical_answer = generate_hypothetical_answer(question)
    if not hypothetical_answer:
        print(f"Failed to generate hypothetical answer for query_id {query_id}")
        return None

    # Embed the hypothetical answer
    max_retries = 5
    for attempt in range(max_retries):
        try:
            hypothetical_embedding = embedding_model.embed_documents([hypothetical_answer])[0]
            break  # Exit loop if successful
        except Exception as e:
            time.sleep(1)  # Optional delay
            if attempt == max_retries - 1:
                print(f"An error occurred while embedding the hypothetical answer: {e}")
                print(f"Retrying embedding ({attempt + 1}/{max_retries})...")
                print(f"Failed to embed the hypothetical answer after {max_retries} attempts.")
                return None
    else:
        return None

    for distance_metric, collection_name in collections.items():
        retrieved_docs, retrieved_metadata = retrieve_documents(hypothetical_embedding, collection_name)
        if not retrieved_docs:
            response = "No answer available"
            retrieved_context = []
        else:
            response = generate_final_answer(question, retrieved_docs)
            retrieved_context = []
            for md, doc in zip(retrieved_metadata, retrieved_docs):
                retrieved_context.append({
                    'doc_id': md.get('doc_id', ''),
                    'text': doc
                })

        result = {
            'query_id': f"{query_id:03d}",
            'query': question,
            'gt_answer': gt_answer,
            'hypothetical_answer': hypothetical_answer,
            'response': response.strip(),
            'retrieved_context': retrieved_context
        }

        results_per_collection[distance_metric] = result

    return results_per_collection

# Initialize result lists
results_hyde_cosine = []
results_hyde_euclidean = []

print("Answering questions using HyDE retrieval with multithreading...")

max_workers = 100  # Adjust based on your system capabilities
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_query_id = {
        executor.submit(process_question, query_id, item): query_id
        for query_id, item in enumerate(chunked_data)
    }

    for future in tqdm(as_completed(future_to_query_id), total=len(future_to_query_id)):
        try:
            results_per_collection = future.result()
            if results_per_collection:
                if 'cosine' in results_per_collection:
                    results_hyde_cosine.append(results_per_collection['cosine'])
                if 'euclidean' in results_per_collection:
                    results_hyde_euclidean.append(results_per_collection['euclidean'])
        except Exception as e:
            query_id = future_to_query_id[future]
            print(f"An error occurred while processing query_id {query_id}: {e}")

print("Saving results...")
if 'cosine' in results_per_collection:
    output_data_hyde_cosine = {'results': results_hyde_cosine}
    with open('rag_results_hyde_cosine.json', 'w', encoding='utf-8') as f:
        json.dump(output_data_hyde_cosine, f, ensure_ascii=False, indent=4)
if 'euclidean' in results_per_collection:
    output_data_hyde_euclidean = {'results': results_hyde_euclidean}
    with open('rag_results_hyde_euclidean.json', 'w', encoding='utf-8') as f:
        json.dump(output_data_hyde_euclidean, f, ensure_ascii=False, indent=4)

print("Done!")
