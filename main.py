from index import  train_bpe_tokenizer;

def main():
    train_bpe_tokenizer; #it will run the training process for the BPE tokenizer


if __name__ == "__main__":
    main()


# This script is designed to train a BPE tokenizer using the provided text file.
# It reads the text, counts word frequencies, trains the tokenizer, and saves the vocabulary and merges to JSON files.
# The script also prints the number of tokens and token IDs for verification.
# Make sure to adjust the file paths and parameters as needed for your specific use case.