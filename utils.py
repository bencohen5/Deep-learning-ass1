# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
def read_data(fname):
    data = []
    f = open(fname,encoding="utf8")
    for line in f.readlines():
        label, text = line.strip().lower().split("\t",1)
        data.append((label, text))
    return data

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data('train')] # train data split to bigrams
DEV   = [(l,text_to_bigrams(t)) for l,t in read_data("dev")] # dev data split to bigrams
TEST   = [(l,text_to_bigrams(t)) for l,t in read_data("test")] # dev data split to bigrams
vec_len =0
for l,f in TEST:
    length = len(f)
    if length>vec_len:
        vec_len=len(f)
print (vec_len)
from collections import Counter
fc = Counter()
for l,feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x,c in fc.most_common(600)])

# label strings to IDs
L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
# feature strings (bigrams) to IDs
F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}
print ("a")
