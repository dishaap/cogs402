from langdetect import detect, DetectorFactory, LangDetectException
import re
import unicodedata as ud

# Fix randomness in langdetect
DetectorFactory.seed = 0  

def preprocess_text(text):
    """Normalize Unicode and remove unwanted characters."""
    text = ud.normalize('NFKC', text)
    text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text)  # Remove URLs and emails
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and symbols
    return text.strip()

def tokenize_words(text):
    return text.split()

def filter_words(text, line_num, total_lines, lang):
    clean_text = preprocess_text(text)
    words = tokenize_words(clean_text)
    kannada_words = []

    for word in words:
        try:
            if detect(word) == lang:  # Keep only words of language
                kannada_words.append(word)
        except LangDetectException:
            continue  # Skip words that can't be detected

    if line_num % 1000 == 0 or line_num == total_lines:  # Print progress every 1000 lines
        print(f"Processed {line_num}/{total_lines} lines...")

    return ' '.join(kannada_words)  # Return the filtered sentence

def read_file(file_path):
    print(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    print(f"Finished reading {len(lines)} lines.")
    return lines

def preprocess_corpus(input_file, output_file, lang):
    """Process corpus and save a new file with only Kannada words."""
    corpus = read_file(input_file)
    total_lines = len(corpus)

    print("Starting text processing...")
    cleaned_corpus = [filter_words(line, i+1, total_lines, lang) for i, line in enumerate(corpus)]

    print(f"Saving cleaned corpus to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in cleaned_corpus:
            if line.strip():  # Avoid empty lines
                f.write(line + '\n')

    print(f"Processing complete! Cleaned corpus saved to {output_file}")
