import os
import psutil

import sys
sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
import codecs
import pickle

from collections import Counter
from tree import Tree as tree


def countSubtree(t):
    return (sum(len(list(x.allNodes())) for x in t.allNodes()))

f = open("/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB3/24/wsj_2415.mrg")
M = 0
for line in f:
    t = tree(string=line)
    l = list(t.allNodes())
    ll = sorted(list(map(countSubtree, l)))
    c = Counter(ll)
    m = (sorted(list(c.items()), key=lambda x: x[1], reverse=True))[0]
    print (m[1], c)
    if m[1] > M:
        M = m[1]


print (M)


