
import json
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import os

# Ensure NLTK data is downloaded
import nltk
nltk.download('punkt')

# File paths (adjust these if your files are named differently)
DOCUMENTS_CSV = './raw/kaggle/documents.csv'
QA_PAIRS1_CSV = './raw/kaggle/multi_passage_answer_questions.csv'       # Replace with actual filename if different
QA_PAIRS2_CSV = './raw/kaggle/single_passage_answer_questions.csv'    # Replace with actual filename if different

# Output file
OUTPUT_JSON = 'chunked_data_kaggle.json'

# Chunking parameters
MAX_TOKENS = 500      # Maximum tokens per chunk
OVERLAP_TOKENS = 50   # Tokens to overlap between chunks

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
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            # Start new chunk with overlap
            if overlap_tokens > 0 and len(current_chunk) >= overlap_tokens:
                overlap = current_chunk[-overlap_tokens:]
            else:
                overlap = current_chunk
            current_chunk = overlap + tokens
            current_tokens = len(current_chunk)
        else:
            current_chunk.extend(tokens)
            current_tokens += token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def main():
    # Check if all required files exist
    required_files = [DOCUMENTS_CSV, QA_PAIRS1_CSV, QA_PAIRS2_CSV]
    for file in required_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Required file '{file}' not found in the working directory.")

    # Load the documents dataset
    docs_df = pd.read_csv(DOCUMENTS_CSV)  # Columns: 'index', 'source_url', 'text'

    # Load and combine the QA pairs datasets
    qa_df1 = pd.read_csv(QA_PAIRS1_CSV)    # Columns: 'document_index', 'question', 'answer'
    qa_df2 = pd.read_csv(QA_PAIRS2_CSV)    # Columns: 'document_index', 'question', 'answer'
    qa_df = pd.concat([qa_df1, qa_df2], ignore_index=True)

    # Ensure that 'document_index' in QA pairs matches 'index' in documents
    if qa_df['document_index'].dtype != docs_df['index'].dtype:
        qa_df['document_index'] = qa_df['document_index'].astype(docs_df['index'].dtype)

    # Create a dictionary for quick lookup of documents by index
    docs_dict = docs_df.set_index('index')['text'].to_dict()

    # Initialize the output list
    output_data = []

    # Iterate over each QA pair and associate with the corresponding document
    for _, qa_row in tqdm(qa_df.iterrows(), total=qa_df.shape[0], desc="Processing QA pairs"):
        doc_index = qa_row['document_index']
        question = qa_row['question']
        answer = qa_row['answer']

        # Retrieve the corresponding document
        if doc_index not in docs_dict:
            print(f"Warning: Document index {doc_index} not found in documents.csv. Skipping QA pair.")
            continue

        context_id = doc_index
        context = docs_dict[doc_index]

        # Chunk the context text
        chunks = chunk_text(context)

        # Create the output item
        output_item = {
            'context_id': context_id,
            'context': context,
            'chunks': chunks,
            'Q': question,
            'A': answer
        }

        output_data.append(output_item)

    # Write the output data to a JSON file
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Preprocessing complete. Output saved to '{OUTPUT_JSON}'.")

if __name__ == "__main__":
    main()
