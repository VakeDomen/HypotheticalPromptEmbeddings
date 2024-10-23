import json
import faiss
import numpy as np
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re

print("Loading chunked data...")
with open('../data/chunked_data.json', 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)

#chunked_data = chunked_data[:10]

print("Loading models...")
embedding_model = OllamaEmbeddings(base_url="hivecore.famnit.upr.si:6666", model='bge-m3')
llm = OllamaLLM(base_url="hivecore.famnit.upr.si:6666", model='mistral-nemo')

print("Generating diverse questions for each chunk...")


texts = []
metadata = []
questions_for_embedding = []
chunks_questions = [] 

def generate_questions(chunk):
    prompt = f"Analyze the input text and generate essential questions that, when answered, capture the main points and core meaning of the text. The questions should be exhaustive and understandable without context. When possible named entities should be referenced by their full name. Only answer with questions where each question should be wirtten in it's own line(separated by newline) with no prefix. Here is the text: \n\n{chunk}\n\nQuestions:"
    questions = llm.invoke(prompt)
    #return [q.strip('- ').strip() for q in questions.strip().split('\n') if q.strip()]
    list_pattern = re.compile(r"^\s*[\-\*\•]|\s*\d+\.\s*|\s*[a-zA-Z]\)\s*|\s*\(\d+\)\s*|\s*\([a-zA-Z]\)\s*|\s*\([ivxlcdm]+\)\s*", re.IGNORECASE)
    
    # Split the response into individual questions and remove list markers using regex
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
        chunks_questions.append(chunk_info)

        for index, question in enumerate(questions):
            questions_for_embedding.append(question)
            texts.append(chunk)
            metadata.append({
                'doc_id': f"x{doc_id}_c{chunk_id}_q{str(index)}",
                'chunk_id': chunk_id,
                'Q': item['Q'],
                'A': item['A']
            })
    except Exception as e:
        print(f"An error occurred while processing chunk {doc_id}-{chunk_id}: {e}")

# Prepare arguments for concurrent processing
chunk_args = []
for doc_id, item in enumerate(chunked_data):
    for chunk_id, chunk in enumerate(item['chunks']):
        chunk_args.append((doc_id, item, chunk_id, chunk))

# Process all chunks concurrently
with ThreadPoolExecutor(max_workers=30) as executor:
    list(tqdm(executor.map(process_chunk, chunk_args), total=len(chunk_args)))

print("Saving chunks and generated questions to 'chunks_with_questions.json'...")
with open('chunks_with_questions.json', 'w', encoding='utf-8') as f:
    json.dump(chunks_questions, f, ensure_ascii=False, indent=4)

print("Embedding generated questions...")

def embed_text(i, text):
    embedding = embedding_model.embed_documents([text])
    return i, embedding[0]  # Since embed_documents returns a list

embeddings = [None] * len(questions_for_embedding)  # Pre-allocate a list for embeddings

print("Embedding in parallel...")
with ThreadPoolExecutor(max_workers=30) as executor:
    futures = {executor.submit(embed_text, i, question): i for i, question in enumerate(questions_for_embedding)}

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
    retrieved_docs = [texts[i] for i in I[0]]  # Retrieve the original chunk
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

max_workers = 30  # Adjust based on your system capabilities
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