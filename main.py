import os
from collections import Counter
from typing import BinaryIO
#from Minbpe.regex import RegexTokenizer

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def count_words_in_chunk(text: str) -> Counter:
    word_freqs = Counter()
    for word in text.strip().split():
        word_bytes = tuple(word.encode("utf-8"))
        word_freqs[word_bytes] += 1
    return word_freqs

def train_bpe_tokenizer(input_path: str, vocab_size: int, special_tokens: list[str]):
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

    # Print number of tokens and token IDs
    print("Number of tokens:", len(global_word_freqs))
    print("Token IDs:", list(global_word_freqs.keys())[:20])  # Print first 20 token IDs

    return tokenizer.vocab, tokenizer.merges

# Example usage:
file_path = "/content/Data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 1000
special_tokens = ["<|endoftext|>"]

vocab, merges = train_bpe_tokenizer(file_path, vocab_size, special_tokens)

print("Vocabulary size:", len(vocab))
print("Number of merges:", len(merges))
print("Merges:", merges)  # Uncomment to see the merges
#print("Vocab:", vocab)  # Uncomment to see the vocabulary




def main():
    print("Hello from bpe!")


if __name__ == "__main__":
    main()

# I will modify the code to ensure it works with the latest changes in Minbpe/regex.py
#when i'am back from school