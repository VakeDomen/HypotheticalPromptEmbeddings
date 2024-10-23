import json
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm


# Load the dataset
dataset = load_dataset('microsoft/ms_marco', "v1.1",  split='train')

dataset = dataset[:]

# Function to split text into chunks
def chunk_text(passage):
    return passage.get("passage_text", [])

# Process each item in the dataset
output_data = []
context_id = 1
answers = dataset.get("answers")
passages = dataset.get("passages")
queries = dataset.get("query")
for itemIndex in tqdm(range(len(answers))):
    Q = queries[itemIndex]
    A = next(iter(answers[itemIndex]), "No answer avalible")
    chunks = passages[itemIndex]["passage_text"]

    output_item = {
        'context_id': context_id,
        'chunks': chunks,
        'Q': Q,
        'A': A
    }
    output_data.append(output_item)
    context_id += 1

# Write output to a JSON file
with open('chunked_data_ms_marco.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
