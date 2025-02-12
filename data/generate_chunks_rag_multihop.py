import json
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import nltk

nltk.download("punkt")

# -----------------------------
# Configuration
# -----------------------------
MAX_TOKENS = 500      # Maximum tokens in each chunk
OVERLAP_TOKENS = 50   # Number of tokens to overlap between consecutive chunks

def chunk_text(text, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS):
    """
    Splits a string `text` into overlapping chunks of up to `max_tokens` each.
    Uses NLTK sentence tokenization, then word tokenization.
    Overlaps the final `overlap_tokens` words between consecutive chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        token_count = len(tokens)

        # If adding this sentence would exceed max_tokens, finalize the current chunk
        if current_len + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))

            # Overlap
            overlap_slice = current_chunk[-overlap_tokens:] if overlap_tokens < len(current_chunk) else current_chunk
            current_chunk = overlap_slice + tokens
            current_len = len(current_chunk)
        else:
            current_chunk.extend(tokens)
            current_len += token_count

    # Add the last chunk if non-empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def main():
    # -----------------------------
    # 1) First, collect all Q&A items (test split)
    # -----------------------------
    mh_test = load_dataset("yixuantt/MultiHopRAG", name="MultiHopRAG", split="train")

    # Build a list of question objects
    # We'll fill the "chunks" later, after we chunk the corpus
    questions = []
    for i, item in enumerate(mh_test):
        questions.append({
            "context_id": i + 1,
            "Q": item["query"],
            "A": item["answer"],
            "chunks": []  # Will be assigned after we chunk the corpus
        })

    # -----------------------------
    # 2) Next, chunk the entire corpus train split
    # -----------------------------
    corpus_train = load_dataset("yixuantt/MultiHopRAG", name="corpus", split="train")

    # Make one big list of chunks from all documents
    all_chunks = []
    for doc in tqdm(corpus_train, desc="Chunking corpus"):
        body_text = doc["body"]
        doc_chunks = chunk_text(body_text, MAX_TOKENS, OVERLAP_TOKENS)
        all_chunks.extend(doc_chunks)

    total_chunks = len(all_chunks)
    num_questions = len(questions)

    # -----------------------------
    # 3) Distribute chunks among question objects
    # -----------------------------
    if num_questions == 0:
        # Edge case: no Q&A found
        # Nothing to distribute, just proceed with saving
        pass
    else:
        # Divide as evenly as possible
        base_chunk_count = total_chunks // num_questions  # integer division
        leftover = total_chunks % num_questions

        # We'll keep track of where we are in all_chunks
        current_index = 0

        for i, q_obj in enumerate(questions):
            # Determine how many chunks to assign to this question
            # For the first `leftover` questions, assign +1 chunk
            chunk_count_for_this = base_chunk_count + (1 if i < leftover else 0)

            # Slice from all_chunks
            assigned = all_chunks[current_index : current_index + chunk_count_for_this]
            current_index += chunk_count_for_this

            # Put them in the question object
            q_obj["chunks"] = assigned

    # -----------------------------
    # 4) Output a single JSON
    # -----------------------------
    # This merges Q & A objects along with the distributed chunks
    output_filename = "combined_data.json"
    print(f"Saving {len(questions)} question objects (with assigned chunks) to '{output_filename}'...")

    with open(output_filename, "w", encoding="utf-8") as f_out:
        json.dump(questions, f_out, ensure_ascii=False, indent=4)

    print("Done!")


if __name__ == "__main__":
    main()
