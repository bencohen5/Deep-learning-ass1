from collections import Counter
import numpy as np


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    e_x = np.exp(x - np.max(x))
    x = e_x / e_x.sum()
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.
    return x


def tanh(x, derivative=False):
    res = np.tanh(x)
    if derivative:
        return 1 - res * res
    return res


def read_data(fname):
    data = []
    f = open(fname, encoding="utf8")
    for line in f.readlines():
        label, text = line.strip().lower().split("\t", 1)
        data.append((label, text))
    return data


def feats_to_vec(features):
    vec = [features.count(c) for c in F2I]
    return np.array(vec)


def feats_to_vec_uni(features):
    vec = [features.count(c) for c in F2I_UNI]
    return np.array(vec)


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


def text_to_unigrams(text):
    return ["%s" % c1 for c1 in text]


TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data('train')]  # train data split to bigrams
DEV = [(l, text_to_bigrams(t)) for l, t in read_data("dev")]  # dev data split to bigrams
TEST = [(l, text_to_bigrams(t)) for l, t in read_data("test")]  # dev data split to bigrams
TRAIN_UNI = [(l, text_to_unigrams(t)) for l, t in read_data('train')]  # train data split to bigrams
DEV_UNI = [(l, text_to_unigrams(t)) for l, t in read_data("dev")]  # dev data split to bigrams
vec_len = 0
for l, f in TEST:
    length = len(f)
    if length > vec_len:
        vec_len = len(f)

fc = Counter()
for l, feats in TRAIN:
    fc.update(feats)
fc_uni = Counter()
for l, feats in TRAIN_UNI:
    fc_uni.update(feats)
# 600 most common bigrams in the training set.
vocab = set([x for x, c in fc.most_common(600)])
# 600 most common unigrams in the training set.
vocab_uni = set([x for x, c in fc.most_common(600)])

# label strings to IDs
L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
# IDs to label strings
I2L = {i: l for l, i in L2I.items()}
# feature strings (bigrams) to IDs
F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}
# feature strings (bigrams) to IDs
F2I_UNI = {f: i for i, f in enumerate(list(sorted(vocab_uni)))}
