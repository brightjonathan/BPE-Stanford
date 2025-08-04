#using tiktoken for tokenization
# This script tokenizes a text file using the tiktoken library, which is designed for.

import tiktoken
import os

# Function to tokenize a file from its path
def tokenize_file_from_path(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    if not file_path.endswith(".txt"):
        print(f"⚠️ Not a .txt file: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, allowed_special={'<|endoftext|>'})  # Fixed parentheses

    print(f"\n📄 File: {file_path}")
    print(f"📚 Vocabulary size: {encoding.n_vocab}")
    print(f"🧮 Number of tokens: {len(tokens)}")
    print(f"🧾 Token IDs (first 20): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")

# Example usage — just change the path below
file_path_1 = "/content/Data/TinyStoriesV2-GPT4-train.txt"
file_path_2 = "/content/sample2.txt"

# Run tokenization for each file
tokenize_file_from_path(file_path_1)
# tokenize_file_from_path(file_path_2)