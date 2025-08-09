import random
from collections import Counter
import json

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def tokenize(sentence):
    return sentence.lower().split()

def build_vocab(tokenized_sentences, min_freq=1):
    counter = Counter()
    for tokens in tokenized_sentences:
        counter.update(tokens)
    vocab = {
        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        '<unk>': 3,
    }
    idx = 4
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = idx
            idx += 1
    return vocab

def tokens_to_ids(tokens, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

def preprocess(source_file, target_file):
    src_sentences = read_file(source_file)
    tgt_sentences = read_file(target_file)

    src_tokens = [tokenize(s) for s in src_sentences]
    tgt_tokens = [tokenize(s) for s in tgt_sentences]

    src_vocab = build_vocab(src_tokens)
    tgt_vocab = build_vocab(tgt_tokens)

    tgt_tokens_in = [['<sos>'] + tokens for tokens in tgt_tokens]
    tgt_tokens_out = [tokens + ['<eos>'] for tokens in tgt_tokens]

    src_ids = [tokens_to_ids(tokens, src_vocab) for tokens in src_tokens]
    tgt_ids_in = [tokens_to_ids(tokens, tgt_vocab) for tokens in tgt_tokens_in]
    tgt_ids_out = [tokens_to_ids(tokens, tgt_vocab) for tokens in tgt_tokens_out]

    return src_ids, tgt_ids_in, tgt_ids_out, src_vocab, tgt_vocab

def train_test_split(src_ids, tgt_ids_in, tgt_ids_out, test_ratio=0.1, seed=42):
    random.seed(seed)
    data = list(zip(src_ids, tgt_ids_in, tgt_ids_out))
    random.shuffle(data)

    n_test = int(len(data) * test_ratio)
    test_data = data[:n_test]
    train_data = data[n_test:]

    src_train, tgt_in_train, tgt_out_train = zip(*train_data)
    src_test, tgt_in_test, tgt_out_test = zip(*test_data)
    return (list(src_train), list(tgt_in_train), list(tgt_out_train)), \
           (list(src_test), list(tgt_in_test), list(tgt_out_test))