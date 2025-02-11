import json
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import nltk
nltk.download('punkt_tab')

# Load the dataset
dataset = load_dataset("explodinggradients/ragas-wikiqa", split='train')

# Process each item in the dataset
output_data = []
context_id = 1
for item in tqdm(dataset):
    chunks = item['context']
    Q = item.get('question', None)
    A = item.get('correct_answer', None)

    # chunks = chunk_text(context)

    output_item = {
        'context_id': context_id,
        'chunks': chunks,
        'Q': Q,
        'A': A
    }
    output_data.append(output_item)
    context_id += 1

# Write output to a JSON file
with open('chunked_data.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
