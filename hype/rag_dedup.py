import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
import numpy as np
import faiss

print("Loading chunked data...")
data = '../data/chunked_data.json'
if len(sys.argv) > 1:
    data = sys.argv[1]
with open(data, 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)

print("Loading models...")
embedding_model = OllamaEmbeddings(base_url="hivecore.famnit.upr.si:6666", model='bge-m3')
llm = OllamaLLM(base_url="hivecore.famnit.upr.si:6666", model='mistral-nemo')

print("Generating diverse questions for each chunk...")

def generate_questions(chunk):
    prompt = (
        "Analyze the input text and generate essential questions that, when answered, capture the main points and core meaning of the text. "
        "The questions should be exhaustive and understandable without context. When possible, named entities should be referenced by their full name. "
        "Only answer with questions where each question should be written in its own line (separated by newline) with no prefix. "
        f"Here is the text: \n\n{chunk}\n\nQuestions:"
    )
    questions = llm.invoke(prompt)
    list_pattern = re.compile(
        r"^\s*[\-\*\•]|\s*\d+\.\s*|\s*[a-zA-Z]\)\s*|\s*\(\d+\)\s*|\s*\([a-zA-Z]\)\s*|\s*\([ivxlcdm]+\)\s*",
        re.IGNORECASE
    )
    return [re.sub(list_pattern, '', q).strip() for q in questions.strip().split('\n') if q.strip()]

def process_chunk(args):
    doc_id, item, chunk_id, chunk = args
    try:
        questions = generate_questions(chunk)
        chunk_info = {
            'doc_id': item['context_id'],
            'chunk_id': chunk_id,
            'chunk_text': chunk,
            'generated_questions': questions
        }
        chunk_data = {
            'chunk_info': chunk_info,
            'texts': [],  # Store the chunk texts for retrieval
            'metadata': [],
            'questions_for_embedding': []  # Store questions for embedding
        }

        for index, question in enumerate(questions):
            # Store the chunk text as the content to be retrieved
            chunk_data['texts'].append(chunk)
            # Metadata includes the chunk text and other info
            chunk_data['metadata'].append({
                'doc_id': f"x{doc_id}_c{chunk_id}_q{str(index)}",
                'chunk_id': chunk_id,
                'Q': item['Q'],
                'A': item['A'],
                'chunk_text': chunk
            })
            # Collect the generated questions for embedding
            chunk_data['questions_for_embedding'].append(question)
        
        return chunk_data
    except Exception as e:
        print(f"An error occurred while processing chunk {doc_id}-{chunk_id}: {e}")
        return None

# Prepare arguments for parallel processing
args_list = [
    (doc_id, item, chunk_id, chunk)
    for doc_id, item in enumerate(chunked_data)
    for chunk_id, chunk in enumerate(item['chunks'])
]

# Initialize lists to collect results
texts = []  # Chunks to retrieve
metadata = []
questions_to_embed = []  # Questions to embed
chunks_questions = []

print("Processing chunks in parallel...")
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_chunk, args) for args in args_list]
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        if result:
            chunks_questions.append(result['chunk_info'])
            texts.extend(result['texts'])  # Collect chunk texts
            metadata.extend(result['metadata'])
            questions_to_embed.extend(result['questions_for_embedding'])  # Collect questions

print("Saving chunks and generated questions to 'chunks_with_questions.json'...")
with open('chunks_with_questions.json', 'w', encoding='utf-8') as f:
    json.dump(chunks_questions, f, ensure_ascii=False, indent=4)

print("Embedding generated questions...")

def embed_text_batch(batch):
    indices, questions_batch = zip(*batch)
    embeddings_batch = embedding_model.embed_documents(questions_batch)
    return list(zip(indices, embeddings_batch))

# Set batch size
batch_size = 5

# Prepare batches for embedding generated questions
embedding_batches = [
    [(i, questions_to_embed[i]) for i in range(j, min(j + batch_size, len(questions_to_embed)))]
    for j in range(0, len(questions_to_embed), batch_size)
]

embeddings = [None] * len(questions_to_embed)  # Pre-allocate a list for embeddings

print("Embedding in parallel...")
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(embed_text_batch, batch) for batch in embedding_batches]
    for future in tqdm(as_completed(futures), total=len(futures)):
        batch_results = future.result()
        for i, embedding in batch_results:
            embeddings[i] = embedding

embeddings = np.array(embeddings).astype('float32')

print("Creating index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def retrieve_documents(question, top_k=30, final_k=5):
    question_embedding = embedding_model.embed_documents([question])
    question_embedding = np.array(question_embedding).astype('float32')
    D, I = index.search(question_embedding, top_k)
    unique_chunks = {}
    for dist, idx in zip(D[0], I[0]):
        md = metadata[idx]
        chunk_text = md['chunk_text']
        # If this chunk_text is not in unique_chunks, or has a smaller distance
        if chunk_text not in unique_chunks or dist < unique_chunks[chunk_text]['distance']:
            unique_chunks[chunk_text] = {
                'distance': dist,
                'metadata': md,
                'chunk_text': chunk_text
            }
    # Sort the unique chunks by their minimal distance
    sorted_chunks = sorted(unique_chunks.values(), key=lambda x: x['distance'])
    # Select the top final_k unique chunks
    top_chunks = sorted_chunks[:final_k]
    retrieved_docs = [item['chunk_text'] for item in top_chunks]
    retrieved_metadata = [item['metadata'] for item in top_chunks]
    return retrieved_docs, retrieved_metadata

def generate_final_answer(question, context):
    context_text = "\n".join(context)
    prompt = f"Answer the following question using the provided context. If no answer can be found in the context, answer 'No answer available'.\n\nContext:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer:"
    final_answer = llm.invoke(prompt)
    return final_answer

results = []

print("Answering questions using Base RAG with multithreading...")

max_workers = 30  # Adjust based on your system capabilities
def process_question(query_id, item):
    question = item['Q']
    gt_answer = item['A']
    if not question or not gt_answer:
        return None

    # Retrieve more hits and deduplicate
    retrieved_docs, retrieved_metadata = retrieve_documents(question, top_k=30, final_k=5)
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
