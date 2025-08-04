import os
from collections import Counter
import json
from Minbpe.regex import RegexTokenizer
from cs366_Stanford.PreTokenization import find_chunk_boundaries, count_words_in_chunk

def train_bpe_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str],
                        vocab_output_path: str, merges_output_path: str,
                        token_ids_output_path: str):
    special_token_str = "<|endoftext|>"
    special_token_bytes = special_token_str.encode("utf-8")
    num_chunks = 8

    global_word_freqs = Counter()

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, special_token_bytes)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_word_freqs = count_words_in_chunk(chunk)
            global_word_freqs.update(chunk_word_freqs)

    tokenizer = RegexTokenizer()
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer.train(text, vocab_size)
    special_token_dict = {token: 256 + len(tokenizer.merges) + i for i, token in enumerate(special_tokens)}
    tokenizer.register_special_tokens(special_token_dict)

    # Save vocabulary to file
    with open(vocab_output_path, "w", encoding="utf-8") as f:
        json.dump({k: v.decode("utf-8", errors="ignore") for k, v in tokenizer.vocab.items()}, f, indent=2)

    # Save merges to file
    with open(merges_output_path, "w", encoding="utf-8") as f:
        json.dump(list(tokenizer.merges), f, indent=2)

    # Tokenize the text
    encoded_tokens = tokenizer.encode(text, allowed_special="all")

    # Save token IDs to file
    with open(token_ids_output_path, "w", encoding="utf-8") as f:
        json.dump(encoded_tokens, f, indent=2)

    # Decode tokens to string representation (for readability)
    decoded_text = tokenizer.decode(encoded_tokens)

    # Print token info
    print("Number of tokens:", len(global_word_freqs))
    print(f"Token IDs (first 40): {encoded_tokens[:40]}")

    return tokenizer.vocab, tokenizer.merges


# === USAGE ===
file_path = "/content/Data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 5000
special_tokens = ["<|endoftext|>"]
vocab_output_path = "/content/bpe_vocab.json"
merges_output_path = "/content/bpe_merges.json"
token_ids_output_path = "/content/token_ids.json"

vocab, merges = train_bpe_tokenizer(
    input_path=file_path,
    vocab_size=vocab_size,
    special_tokens=special_tokens,
    vocab_output_path=vocab_output_path,
    merges_output_path=merges_output_path,
    token_ids_output_path=token_ids_output_path,
)

print("Vocabulary size:", len(vocab))
print("Number of merges:", len(merges))
print("Merges:", merges)  # Uncomment to see the merges
print("Vocab:", vocab)  # Uncomment to see the vocabulary



# This script is designed to train a BPE tokenizer using the provided text file.
# It reads the text, counts word frequencies, trains the tokenizer, and saves the vocabulary and merges to JSON files.
# The script also prints the number of tokens for verification.
# Make sure to adjust the file paths and parameters as needed for your specific use case.