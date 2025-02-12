import json
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm


# Load the dataset
dataset = load_dataset('portkey/truthful_qa_context', split='train')

# Chunking parameters
MAX_TOKENS = 500  # Maximum tokens per chunk
OVERLAP_TOKENS = 50  # Tokens to overlap between chunks

# Function to split text into chunks
def chunk_text(text, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        token_count = len(tokens)

        if current_tokens + token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            # Start new chunk with overlap
            overlap = current_chunk[-overlap_tokens:] if overlap_tokens < len(current_chunk) else current_chunk
            current_chunk = overlap + tokens
            current_tokens = len(current_chunk)
        else:
            current_chunk.extend(tokens)
            current_tokens += token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Process each item in the dataset
output_data = []
context_id = 1
for item in tqdm(dataset):
    context = item['context']
    Q = item.get('question', None)
    A = item.get('best_answer', None)

    chunks = chunk_text(context)

    output_item = {
        'context_id': context_id,
        'context': context,
        'chunks': chunks,
        'Q': Q,
        'A': A
    }
    output_data.append(output_item)
    context_id += 1

# Write output to a JSON file
with open('chunked_data_truthful_qa.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
