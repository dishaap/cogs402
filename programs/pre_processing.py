from langdetect import detect, DetectorFactory, LangDetectException
import numpy as np
import re
import unicodedata as ud

def preprocess_text(text):
    # Normalize Unicode
    text = ud.normalize('NFKC', text)
    # Remove URLs, emails, and special characters
    text = re.sub(r'http\S+|www\S+|\S+@\S+', '', text)  # Remove links and emails
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and symbols
    return text.strip()

def tokenize_words(text):
    return text.split()

def is_mixed_language(text):
    clean_text = preprocess_text(text)
    words = tokenize_words(clean_text)
    non_kn_count, kn_count = 0, 0
    
    for word in words:
        try:
            lang = detect(word)
            if lang != 'kn':
                non_kn_count += 1
            else:
                kn_count += 1
        except LangDetectException:
            continue  # Skip words that can't be detected
    
    return non_kn_count > 0 and kn_count > 0  # True if both English and another language are pr

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

file_path = '../data/kn-ta/MultiHPLT.kn.txt'  # Change to the actual file path
corpus = read_file(file_path)
mixed = np.zeros(len(corpus))

# Detect mixed-language lines
for line in corpus:
    if is_mixed_language(line):
        mixed[corpus.index(line)] = 1

print(mixed)