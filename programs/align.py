# alignment of ta-kn and kn-en parallel corpora
from pre_processing import preprocess_corpus

def load_corpus(file1, file2):
    print(f"Reading corpus: {file1} and {file2}...")
    
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        source_sentences = f1.readlines()
        target_sentences = f2.readlines()
    
    corpus_size = len(source_sentences)
    print(f"Loaded {corpus_size} sentence pairs from {file1} and {file2}.")
    
    return list(zip(source_sentences, target_sentences))

def find_common_sentences(en_ka, tn_ka):
    """Find common Kannada sentences with progress tracking."""
    print("Processing alignment...")

    # Convert to dictionary for fast lookup
    en_ka_dict = {ka.strip(): en.strip() for en, ka in en_ka}
    tn_ka_dict = {ka.strip(): tn.strip() for tn, ka in tn_ka}

    # Find common Kannada sentences
    common_ka_sentences = set(en_ka_dict.keys()) & set(tn_ka_dict.keys())
    total_common = len(common_ka_sentences)
    print(f"Found {total_common} common Kannada sentences.")

    # Extract aligned sentences with progress tracking
    common_data = []
    for i, ka in enumerate(common_ka_sentences):
        common_data.append((en_ka_dict[ka], ka, tn_ka_dict[ka]))
        if (i + 1) % 1000 == 0 or i + 1 == total_common:
            print(f"Aligned {i+1}/{total_common} sentences...")

    return common_data

def save_common_sentences(common_data, output_file):
    """Save aligned sentences with progress tracking."""
    print(f"Saving aligned corpus to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for en, ka, tn in common_data:
            f.write(f"{en}\t{ka}\t{tn}\n")

    print(f"Alignment complete! Saved {len(common_data)} aligned sentence pairs.")

pre_knta_kn = "../data/kn-ta/NLLB.kn-ta.kn"
pre_knta_ta = "../data/kn-ta/NLLB.kn-ta.ta"
pre_knen_kn = "../data/kn-en/NLLB.en-kn.ka"
pre_knen_en = "../data/kn-en/NLLB.en-kn.en"

post_knta_kn = "../data/clean_kn-ta/kn-ta.kn"
post_knta_ta = "../data/clean_kn-ta/kn-ta.ta"
post_knen_kn = "../data/clean_kn-en/en-kn.ka"
post_knen_en = "../data/clean_kn-en/en-kn.en"

preprocess_corpus(pre_knta_kn, post_knta_kn, "kn")
preprocess_corpus(pre_knta_ta, post_knta_ta, "ta")
preprocess_corpus(pre_knen_kn, post_knen_kn, "kn")
preprocess_corpus(pre_knen_en, post_knen_en, "en")

# Load the corpora
en_ka = load_corpus(post_knen_en, post_knen_kn)  # English-Kannada
tn_ka = load_corpus(post_knta_ta, post_knta_kn)  # Tamil-Kannada

common_data = find_common_sentences(en_ka, tn_ka)

save_common_sentences(common_data, "common_sentences.txt")