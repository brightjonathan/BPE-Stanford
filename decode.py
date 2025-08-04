import json

# === Step 1: Load vocab from JSON ===
def load_vocab(vocab_path: str) -> dict[int, bytes]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)

    # Convert keys to int and values to bytes
    vocab = {}
    for k, v in raw_vocab.items():
        try:
            token_bytes = v.encode("utf-8")
        except Exception:
            token_bytes = repr(v).encode("utf-8")
        vocab[int(k)] = token_bytes

    return vocab

# === Step 2: Decode token IDs into string ===
def decode_token_ids(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    decoded_bytes = b""

    for token_id in token_ids:
        token_bytes = vocab.get(token_id)
        if token_bytes is None:
            print(f"[Warning] Token ID {token_id} not found in vocab.")
            continue
        decoded_bytes += token_bytes

    try:
        return decoded_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        print("⚠️ Unicode decode error:", e)
        return decoded_bytes.decode("utf-8", errors="replace")


# === Step 3: Define your token IDs NOTE: Few tokens for texting ===
token_ids = [
    369, 373, 256, 375, 376, 276, 256, 443, 445, 379,
    292, 46, 292, 380, 278, 449, 265, 451, 453, 454,
    46, 311, 342, 456, 312, 458, 44, 459, 390, 460,
    347, 462, 463, 465, 348, 256, 349, 46, 466, 392
]

# === Step 4: File path to vocab ===
vocab_path = "/content/bpe_vocab.json"  # <-- Update this if needed

# === Step 5: Load vocab and decode ===
vocab = load_vocab(vocab_path)
decoded_text = decode_token_ids(token_ids, vocab)

# === Step 6: Output ===
print("🧾 Decoded Text:")
print(decoded_text)

# Note: Ensure the vocab_path points to the correct JSON file containing the vocabulary.
# This code assumes the vocabulary is in the format {token_id: token_string}.
# The token IDs should be integers, and the tokens should be UTF-8 encoded strings.
# The decoded text will be printed at the end.
# This script is designed to decode a list of token IDs into a human-readable string using a vocabulary loaded from a JSON file.
# This script is designed to train a BPE tokenizer using the provided text file.