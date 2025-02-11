import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import time
from langchain_ollama.llms import OllamaLLM

print("Loading chunked data...")
data = '../data/chunked_data.json'
if len(sys.argv) > 1:
    data = sys.argv[1]
with open(data, 'r', encoding='utf-8') as f:
    chunked_data = json.load(f)

print("Loading model...")
llm = OllamaLLM(model='mistral-nemo')

def generate_questions(chunk):
    prompt = (
        "Analyze the input text and generate essential questions that, when answered, capture the main points and core meaning of the text. "
        "The questions should be exhaustive and understandable without context. When possible, named entities should be referenced by their full name. "
        "Only answer with questions where each question should be written in its own line (separated by newline) with no prefix. "
        f"Here is the text: \n\n{chunk}\n\nQuestions:"
    )
    max_retries = 10
    final_questions = []

    for attempt in range(max_retries):
        try:
            questions = llm.invoke(prompt)
            list_pattern = re.compile(
                r"^\s*[\-\*\•]|\s*\d+\.\s*|\s*[a-zA-Z]\)\s*|\s*\(\d+\)\s*|\s*\([a-zA-Z]\)\s*|\s*\([ivxlcdm]+\)\s*",
                re.IGNORECASE
            )
            final_questions = [re.sub(list_pattern, '', q).strip() for q in questions.strip().split('\n') if q.strip()]
            
            if len(final_questions) > 1:
                return final_questions

        except Exception as e:
            if attempt + 1 == max_retries:
                print(f"An error occurred while invoking LLM: {e}. Retrying ({attempt + 1}/{max_retries})...")
            if attempt == max_retries - 1:
                return []
            time.sleep(5)
    return []

    
    return questions

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
        return chunk_info
    except Exception as e:
        print(f"An error occurred while processing chunk {doc_id}-{chunk_id}: {e}")
        return None

# Prepare arguments for parallel processing
args_list = [
    (doc_id, item, chunk_id, chunk)
    for doc_id, item in enumerate(chunked_data)
    for chunk_id, chunk in enumerate(item['chunks'])
]

chunks_questions = []

print("Generating questions in parallel...")

try:
    with ThreadPoolExecutor(max_workers=150) as executor:
        futures = [executor.submit(process_chunk, args) for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result(timeout=300)  # Timeout after 5 minutes
            if result:
                chunks_questions.append(result)
except KeyboardInterrupt:
    print("\nInterrupted by user. Saving collected results...")
except TimeoutError:
    print("A task timed out and was skipped.")
except Exception as e:
    print(f"An error occurred during processing: {e}")

print("Saving chunks and generated questions to 'chunks_with_questions.json'...")
with open('chunks_with_questions.json', 'w', encoding='utf-8') as f:
    json.dump(chunks_questions, f, ensure_ascii=False, indent=4)

print("Done!")
