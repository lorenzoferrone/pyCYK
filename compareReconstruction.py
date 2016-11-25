import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
import sys


if sys.platform == 'darwin': # sto sul mio pc
    sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
else: #sto deployando su venere
    sys.path.append("/home/ferrone/pyDTK_sync/src")
import random
import pickle
import time
import gc
import seaborn as sns


import itertools

from tree import Tree as tree
import grammar
import dtk
import dtk2
import operation as op
from metrics import fscore, precision, recall, labeled_fscore, labeled_precision, labeled_recall
import treeToCYKMatrix
from loadPennTree import loadPennTree

from runExp import distort


def cos(a, b):
    return 1 - (numpy.dot(a, b)/(numpy.linalg.norm(a)*numpy.linalg.norm(b)))

def dist(a, b):
    return numpy.linalg.norm(b-a)

input_ = "/home/ferrone/dtknn/BinaryMatrice/8192.0.6/pennTreeMatrix.section.23.numberified.npy"
inputs = numpy.load(input_)

# reconstructed = numpy.load("/home/ferrone/dtknn/KerasVersion/reconstructed_number_lstm.npy")
# reconstructed = numpy.load("/home/ferrone/dtknn/KerasVersion/reconstructed_from_encoded.npy")
# reconstructed = numpy.load("/home/ferrone/dtknn/encoded/test_8192_06_1_sem.npy")
# reconstructed = numpy.load("/home/ferrone/dtknn/EXPE/8192_512_512/test_pred.512.512.1.npy")
# reconstructed = numpy.load("/home/ferrone/dtknn/KerasVersion/pseudoInverse_prediction.npy")
# reconstructed = numpy.load('/home/ferrone/dtknn/KerasVersion/reconstructed_from_encoded_numberified.npy')
reconstructed = numpy.load('/home/ferrone/dtknn/KerasVersion/semantic_reconstruced.npy')

distorted = [distort(x, 10) for x in inputs]
print (inputs[0])
print (distorted[0])

coss = [cos(b, a) for (a, b) in zip(inputs, distorted)]
dists = [dist(b, a) for (a, b) in zip(inputs, distorted)]
print ("cosine original-distorted:", numpy.mean(coss), numpy.var(coss))
print ("distance original-distorted:", numpy.mean(dists), numpy.var(dists))

inputs_norm = [numpy.linalg.norm(a) for a in inputs]
print ("norm original:", numpy.mean(inputs_norm), numpy.var(inputs_norm))

rec_norm = [numpy.linalg.norm(a) for a in distorted]
print ("norm distorted:", numpy.mean(rec_norm), numpy.var(rec_norm))


print ("########")

coss = [cos(b, a) for (a, b) in zip(inputs, reconstructed)]
dists = [dist(b, a) for (a, b) in zip(inputs, reconstructed)]
print ("cosine original-reconstructed:", numpy.mean(coss), numpy.var(coss))
print ("distance original-reconstructed:", numpy.mean(dists), numpy.var(dists))

inputs_norm = [numpy.linalg.norm(a) for a in inputs]
print ("norm original:", numpy.mean(inputs_norm), numpy.var(inputs_norm))

rec_norm = [numpy.linalg.norm(a) for a in reconstructed]
print ("norm reconstructed:", numpy.mean(rec_norm), numpy.var(rec_norm))

print ("########")

nl = []
for i, r in zip(inputs, reconstructed):
    n = numpy.linalg.norm(i)
    r = r/numpy.linalg.norm(r)
    r = r*n
    nl.append(r)

dists = [dist(b, a) for (a, b) in zip(inputs, nl)]
print ('distance original-reconstructed-renormalized:', numpy.mean(dists), numpy.var(dists))
