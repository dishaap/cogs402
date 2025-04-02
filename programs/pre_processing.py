from langdetect import detect, DetectorFactory, LangDetectException
import re
import unicodedata as ud

# Fix randomness in langdetect
DetectorFactory.seed = 0  

def preprocess_text(text):
    """Normalize Unicode and remove unwanted characters."""
    text = ud.normalize('NFKD', text)
    text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text)  # Remove URLs and emails

    """This removes vowel symbols so commented out for now."""
    # text = re.sub(r'\?', '', text)  # Remove punctuation and symbols
    return text.strip()

def tokenize_words(text):
    return text.split()

def filter_words(text, lang):
    clean_text = preprocess_text(text)
    words = tokenize_words(clean_text)
    kannada_words = []

    for word in words:
        try:
            if detect(word) == lang:  # Keep only words of language
                kannada_words.append(word)
        except LangDetectException:
            continue  # Skip words that can't be detected

    return ' '.join(kannada_words) # Return the filtered sentence

def read_file(file_path):
    print(f"Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    print(f"Finished reading {len(lines)} lines.")
    return lines

def preprocess_corpus(input_file, output_file, lang):
    """Process corpus and save a new file with only Kannada words."""
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        total_lines = sum(1 for _ in infile)  # Count total lines
        infile.seek(0)  # Reset file pointer to beginning

        for i, line in enumerate(infile, 1):
            cleaned_line = filter_words(line, lang)
            if cleaned_line.strip():  # Avoid writing empty lines
                outfile.write(cleaned_line + '\n')

            if i % 1000 == 0 or i == total_lines:  # Show progress every 1000 lines
                print(f"Processed {i}/{total_lines} lines...")

    print(f"Processing complete! Cleaned corpus saved to {output_file}")


"""Test"""
# input_file = "../data/test_kn.kn"
# output_file = "../data/test_cleaned_kn.kn"
# preprocess_corpus(input_file, output_file, 'kn')