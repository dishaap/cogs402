# LAB 3 from Low Resource Lang Labs

import conllu
import os
import torch
import torch.nn as nn
import numpy as np
import re
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import log_softmax, relu
from torch.optim import Adam, SGD
from itertools import islice
from collections import namedtuple
from random import random, seed, shuffle

UNK="<unk>"
START="<start>"
END="<end>"
PAD="<pad>"

word_transform = lambda s: [word_vocab[w] for w in ["<start>"] + s + ["<end>"]]
pos_transform = lambda s: [pos_vocab[w] for w in s]
char_transform = lambda w: [char_vocab[c] for c in ["<start>"] + w + ["<end>"]]

Tagged_Data = namedtuple("Example",["word", "pos", "char"])

train_file_path = "../data/ud/ta_train.conllu"
dev_file_path = "../data/ud/ta_dev.conllu"
test_file_path = "../data/ud/ta_test.conllu"

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

    def read_data(dire, lang): # make this dynamic using os
        train_data = conllu.parse(open(os.path.join(dire, f"{lang}_train")).read())
        test_data = conllu.parse(open(os.path.join(dire, f"{lang}_test")).read())
        dev_data = conllu.parse(open(os.path.join(dire, f"{lang}_dev")).read())
        return train_data, test_data, dev_data
    
    train_data, test_data, dev_data = read_data("data/ud", lan)
    train = UDDataset(train_data)
    test = UDDataset(test_data)
    dev = UDDataset(dev_data)

    word_vocab = char_vocab = pos_vocab = None
    
    if vocabs:
        word_vocab, char_vocab, pos_vocab = vocabs
    else:

        # building word vocab using torchtext, throws error, debug
        word_vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=["<unk>", "<start>", "<end>"])
        word_vocab.set_default_index(word_vocab["<unk>"])
        
        char_vocab = build_vocab_from_iterator(yield_chars(train_data), specials=["<unk>", "<start>", "<end>", "<pad>"])
        char_vocab.set_default_index(word_vocab["<unk>"])
        
        pos_vocab = build_vocab_from_iterator(yield_pos(train_data), specials=["<unk>"])
        pos_vocab.set_default_index(word_vocab["<unk>"])
    
    def split_char_sequence(chars, tokens):
        word_lens = [len(w) for w in tokens]
        chars = iter(chars)
        return [list(islice(chars, elem)) for elem in word_lens]
    
    # collate batches using tensors
    def collate_batch(batch):
        pos_list, token_list, char_list, word_lens = [], [], [], []
        for tokens, chars, pos in zip(yield_tokens(batch), yield_chars(batch), yield_pos(batch)):
            token_tensor = torch.tensor(word_transform(tokens), dtype=torch.long).unsqueeze(1)
            pos_tensor = torch.tensor(pos_transform(pos), dtype=torch.long).unsqueeze(1)
            chars = split_char_sequence(chars, tokens)
            chars = [char_transform(w) for w in chars]
            char_tensors = [torch.tensor(cs, dtype=torch.long) for cs in chars]

            pos_list.append(pos_tensor)
            token_list.append(token_tensor)
            char_list += char_tensors

        # return as tuple

        return Tagged_Data(token_list[0],
                       pos_list[0],
                       (pad_sequence(char_list, batch_first=True, padding_value=char_vocab["<pad>"]).unsqueeze(0),
                        len(token_list[0])-2,
                        [len(w) for w in tokens]))

    test_iter = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_batch)
    dev_iter = DataLoader(dev_data, batch_size=1, shuffle=False, collate_fn=collate_batch)
    train_iter = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_batch)
    
    return train_iter, dev_iter, test_iter, word_vocab, char_vocab, pos_vocab

train_iter, dev_iter, test_iter, word_vocab, char_vocab, pos_vocab = read_ud_data("ta")

""" print("Count of word types:", len(word_vocab))
print("Count of character types:", len(char_vocab))
print("Count of POS types:", len(pos_vocab)) """

# Simple BiLSTM POS-tagger
# Ensure reproducible results.
seed(0)
torch.manual_seed(0)
np.random.seed(0)

# Hyperparameters
EMBEDDING_DIM=50
RNN_HIDDEN_DIM=50
RNN_LAYERS=1
BATCH_SIZE=10
EPOCHS=5


# feed forward processing for a sentence tagger, to be wrapped by POStagger
class BidirectionalLSTM(nn.Module):
    def __init__(self):
        super(BidirectionalLSTM,self).__init__()
        self.forward_rnn = nn.LSTM(EMBEDDING_DIM, RNN_HIDDEN_DIM, RNN_LAYERS)
        self.backward_rnn = nn.LSTM(EMBEDDING_DIM, RNN_HIDDEN_DIM, RNN_LAYERS)
        
    def forward(self,sentence):
        fwd_hss, _ = self.forward_rnn(sentence)
        bwd_hss, _ = self.backward_rnn(sentence.flip(0))
        return torch.cat([fwd_hss, bwd_hss.flip(0)], dim=2)
        
def drop_words(sequence,word_dropout):
    seq_len, _ = sequence.size()
    dropout_sequence = sequence.clone()
    for i in range(1,seq_len-1):
        if random() < word_dropout:
            dropout_sequence[i,0] = word_vocab[UNK]
    return dropout_sequence
        
class SentenceEncoder(nn.Module):
    def __init__(self):
        super(SentenceEncoder,self).__init__()

        self.vocabulary = word_vocab
        self.embedding = nn.Embedding(len(self.vocabulary),EMBEDDING_DIM)
        self.rnn = BidirectionalLSTM()
        
    def forward(self,ex,word_dropout):
        embedded = self.embedding(drop_words(ex.word,word_dropout))
        hss = self.rnn(embedded)
        return hss[1:-1]
        
class FeedForward(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(FeedForward, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(input_dim,input_dim)
        self.linear2 = nn.Linear(input_dim,output_dim)
        
    def forward(self,tensor):
        layer1 = relu(self.linear1(tensor))
        layer2 = self.linear2(layer1).log_softmax(dim=2)
        return layer2