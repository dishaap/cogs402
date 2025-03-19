# LAB 3 from Low Resource Lang Labs

import conllu
import torch
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from itertools import islice
from collections import namedtuple

UNK="<unk>"
START="<start>"
END="<end>"
PAD="<pad>"

word_transform = lambda s: [word_vocab[w] for w in ["<start>"] + s + ["<end>"]]
pos_transform = lambda s: [pos_vocab[w] for w in s]
char_transform = lambda w: [char_vocab[c] for c in ["<start>"] + w + ["<end>"]]

Example = namedtuple("Example",["word", "pos", "char"])

PAD="<pad>"
UNK="<unk>"
START="<start>"
END="<end>"

train_file_path = "../data/ud/ta_ttb-ud-train.conllu"
test_file_path = "../data/ud/ta_ttb-ud-test.conllu"

class UDDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
def yield_tokens(data):
    for ex in data:
        yield([tok["form"] for tok in ex])
        
def yield_chars(data):
    for ex in data:
        yield([c for tok in ex for c in tok["form"]])
        
def yield_pos(data):
    for ex in data:
        yield([tok["upos"] for tok in ex])
    
# parse from conllu files
def read_ud_data(lan, vocabs = None):

    def read_data(train_file_path, test_file_path): # make this dynamic using os
        train_data = conllu.parse(open(train_file_path).read())
        test_data = conllu.parse(open(test_file_path).read())
        return train_data, test_data
    
    train_data, test_data = read_data(train_file_path, test_file_path)
    train = UDDataset(train_data)
    test = UDDataset(test_data)

    word_vocab = char_vocab = pos_vocab = None
    
    if vocabs:
        word_vocab, char_vocab, pos_vocab = vocabs
    else:
        word_vocab = build_vocab_from_iterator(yield_tokens(train_data),
                                               specials=["<unk>", "<start>", "<end>"])
        word_vocab.set_default_index(word_vocab["<unk>"])
        char_vocab = build_vocab_from_iterator(yield_chars(train_data),
                                               specials=["<unk>", "<start>", "<end>", "<pad>"])
        char_vocab.set_default_index(word_vocab["<unk>"])
        pos_vocab = build_vocab_from_iterator(yield_pos(train_data),
                                               specials=["<unk>"])
        pos_vocab.set_default_index(word_vocab["<unk>"])
    
    def split_char_sequence(chars, tokens):
        word_lens = [len(w) for w in tokens]
        chars = iter(chars)
        return [list(islice(chars, elem)) for elem in word_lens]
    
    def collate_batch(batch):
        pos_list, token_list, char_list, word_lens = [], [], [], []
        for tokens, chars, pos in zip(yield_tokens(batch), 
                                      yield_chars(batch),
                                      yield_pos(batch)):
            # Your code here
            token_tensor = torch.tensor(word_transform(tokens), dtype=torch.long).unsqueeze(1)
            pos_tensor = torch.tensor(pos_transform(pos), dtype=torch.long).unsqueeze(1)
            chars = split_char_sequence(chars, tokens)
            chars = [char_transform(w) for w in chars]
            char_tensors = [torch.tensor(cs, dtype=torch.long) for cs in chars]

            pos_list.append(pos_tensor)
            token_list.append(token_tensor)
            char_list += char_tensors

        return Example(token_list[0],
                       pos_list[0],
                       (pad_sequence(char_list, batch_first=True, padding_value=char_vocab["<pad>"]).unsqueeze(0),
                        len(token_list[0])-2,
                        [len(w) for w in tokens]))

    test_iter = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_batch)
    dev_iter = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_batch)
    train_iter = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_batch)
    
    return train_iter, dev_iter, test_iter, word_vocab, char_vocab, pos_vocab