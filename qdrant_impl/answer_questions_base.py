import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

# Import Qdrant Client
from qdrant_client import QdrantClient

# Collections mapping
MAX_RETRIES = 10
COLLECTION_PREFIX="ragbench_"
collections = {
    "base_cosine": f"{COLLECTION_PREFIX}chunks_cosine",
    "base_euclidean": f"{COLLECTION_PREFIX}chunks_euclidean",
    "hype_cosine": f"{COLLECTION_PREFIX}questions_cosine",
    "hype_euclidean": f"{COLLECTION_PREFIX}questions_euclidean",
    "hype_cosine_dedup": f"{COLLECTION_PREFIX}questions_cosine",
    "hype_euclidean_dedup": f"{COLLECTION_PREFIX}questions_euclidean"
}

print("Loading chunked data to retrieve Q and A for answering...")
data = "../data/chunked_data.json"
if len(sys.argv) > 1:
    data = sys.argv[1]
with open(data, "r", encoding="utf-8") as f:
    chunked_data = json.load(f)


print("Loading models...")
embedding_model = OllamaEmbeddings(base_url="hivecore.famnit.upr.si:6666", model="bge-m3")
llm = OllamaLLM(base_url="hivecore.famnit.upr.si:6666", model="mistral-nemo")

print("Connecting to Qdrant...")
client = QdrantClient(host="localhost", port=6333)

def retrieve_documents(question, collection_name, top_k=5, deduplicate=False):
    max_retries = MAX_RETRIES
    for attempt in range(max_retries):
        try:
            question_embedding = embedding_model.embed_documents([question])[0]
            break  # Exit loop if successful
        except Exception as e:
            time.sleep(1)  # Optional: add delay between retries
            if attempt == max_retries - 1:
                print(f"An error occurred while embedding the question: {e}")
                print(f"Retrying embedding ({attempt + 1}/{max_retries})...")
                print(f"Failed to embed the question after {max_retries} attempts.")
                return [], []


    for attempt in range(max_retries):
        try:
            search_result = client.search(
                collection_name=collection_name,
                query_vector=question_embedding,
                limit=top_k * 5 if deduplicate else top_k,  
            )
            break  
        except Exception as e:
            time.sleep(1)  # Optional: add delay between retries
            if attempt == max_retries - 1:
                print(f"An error occurred during Qdrant search: {e}")
                print(f"Retrying search ({attempt + 1}/{max_retries})...")
                print(f"Failed to search Qdrant after {max_retries} attempts.")
                return [], []
    
    retrieved_docs = []
    retrieved_metadata = []
    seen_points = set()

    for hit in search_result:
        id = str(hit.payload["doc_id"]) + "_"  + str(hit.payload["chunk_id"])
        if deduplicate and id in seen_points:
            continue  # Skip duplicate

        seen_points.add(id)

        # For HyPE collections, retrieve the original chunk text
        retrieved_docs.append(hit.payload["text"])
        retrieved_metadata.append(hit.payload)

        if len(retrieved_docs) >= top_k:
            break  # Stop after collecting top_k unique documents

    return retrieved_docs[:top_k], retrieved_metadata[:top_k]

def generate_final_answer(question, context):
    max_retries = MAX_RETRIES
    context_text = "\n".join(context)
    prompt = f"Answer the following question using the provided context. \n\nContext:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer:"
    for attempt in range(max_retries):
        try:
            final_answer = llm.invoke(prompt)
            break  # Exit loop if successful
        except Exception as e:
            # print(prompt)
            
            time.sleep(1)  # Optional: add delay between retries
            if attempt == max_retries - 1:
                print(f"An error occurred while invoking the LLM: {e}")
                print(f"Retrying LLM invocation ({attempt + 1}/{max_retries})...")  
                print(f"Failed to invoke LLM after {max_retries} attempts.")
                return "No answer available"
    else:
        return "No answer available"

    return final_answer.strip()

def process_question(query_id, item):
    question = item["Q"]
    gt_answer = item["A"]
    if not question or not gt_answer:
        return None

    results_per_collection = {}

    for key, collection_name in collections.items():
        deduplicate = "dedup" in key  

        retrieved_docs, retrieved_metadata = retrieve_documents(
            question, collection_name, deduplicate=deduplicate
        )

        if not retrieved_docs:
            response = "No answer available"
            retrieved_context = []
        else:
            response = generate_final_answer(question, retrieved_docs)
            retrieved_context = []
            for md, doc in zip(retrieved_metadata, retrieved_docs):
                retrieved_context.append({
                    "doc_id": md.get("doc_id", ""),
                    "text": doc
                })

        result = {
            "query_id": f"{query_id:03d}",
            "query": question,
            "gt_answer": gt_answer,
            "response": response.strip(),
            "retrieved_context": retrieved_context
        }

        results_per_collection[key] = result

    return results_per_collection

# Initialize result lists
results_base_cosine = []
results_base_euclidean = []
results_hype_cosine = []
results_hype_euclidean = []
results_hype_cosine_dedup = []
results_hype_euclidean_dedup = []

print("Answering questions...")

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
                # Base results
                if "base_cosine" in results_per_collection:
                    results_base_cosine.append(results_per_collection["base_cosine"])
                if "base_euclidean" in results_per_collection:
                    results_base_euclidean.append(results_per_collection["base_euclidean"])

                # HyPE results with duplicates
                if "hype_cosine" in results_per_collection:
                    results_hype_cosine.append(results_per_collection["hype_cosine"])
                if "hype_euclidean" in results_per_collection:
                    results_hype_euclidean.append(results_per_collection["hype_euclidean"])

                if "hype_cosine_dedup" in results_per_collection:
                    results_hype_cosine_dedup.append(results_per_collection["hype_cosine_dedup"])
                if "hype_euclidean_dedup" in results_per_collection:
                    results_hype_euclidean_dedup.append(results_per_collection["hype_euclidean_dedup"])

                # HyPE results with deduplication (processed within process_question)
                # We need to process deduplicated results separately
                # Modify process_question to return deduplicated results separately if needed
        except Exception as e:
            query_id = future_to_query_id[future]
            print(f"An error occurred while processing query_id {query_id}: {e}")

print("Saving results...")
# Save base results
output_data_base_cosine = {"results": results_base_cosine}
with open("rag_results_base_cosine.json", "w", encoding="utf-8") as f:
    json.dump(output_data_base_cosine, f, ensure_ascii=False, indent=4)

output_data_base_euclidean = {"results": results_base_euclidean}
with open("rag_results_base_euclidean.json", "w", encoding="utf-8") as f:
    json.dump(output_data_base_euclidean, f, ensure_ascii=False, indent=4)

# Save HyPE results with duplicates
output_data_hype_cosine = {"results": results_hype_cosine}
with open("rag_results_hype_cosine.json", "w", encoding="utf-8") as f:
    json.dump(output_data_hype_cosine, f, ensure_ascii=False, indent=4)

output_data_hype_euclidean = {"results": results_hype_euclidean}
with open("rag_results_hype_euclidean.json", "w", encoding="utf-8") as f:
    json.dump(output_data_hype_euclidean, f, ensure_ascii=False, indent=4)

output_data_hype_cosine_dedup = {"results": results_hype_cosine_dedup}
with open("rag_results_hype_cosine_dedup.json", "w", encoding="utf-8") as f:
    json.dump(output_data_hype_cosine_dedup, f, ensure_ascii=False, indent=4)

output_data_hype_euclidean_dedup = {"results": results_hype_euclidean_dedup}
with open("rag_results_hype_euclidean_dedup.json", "w", encoding="utf-8") as f:
    json.dump(output_data_hype_euclidean_dedup, f, ensure_ascii=False, indent=4)

print("All results saved with and without deduplication.")
