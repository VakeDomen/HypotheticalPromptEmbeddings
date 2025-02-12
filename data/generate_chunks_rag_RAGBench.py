import json
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

# Load the dataset

# Function to split text into chunks
def chunk_text(passage):
    return passage.get("passage_text", [])

subsets = ['covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa']

output_data = []
context_id = 1

for subset in subsets:
    dataset = load_dataset('rungalileo/ragbench', subset, split='train')
    dataset = dataset[:]
    answers = dataset.get("response")
    passages = dataset.get("documents")
    queries = dataset.get("question")
    for itemIndex in tqdm(range(len(answers))):
        Q = queries[itemIndex]
        A = answers[itemIndex]
        chunks = passages[itemIndex]

        output_item = {
            'context_id': context_id,
            'chunks': chunks,
            'Q': Q,
            'A': A
        }
        output_data.append(output_item)
        context_id += 1

with open('chunked_data_ragbench.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
