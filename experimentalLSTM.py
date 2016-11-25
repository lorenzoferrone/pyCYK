import pickle
import sys
sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
import dtk
import tree
from loadPennTree import loadPennTree
import grammar
import numpy

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, RepeatVector
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.layers.recurrent import LSTM, GRU

def tokenizeTree(stringTree):
    tree_ = tree.Tree(string=stringTree)
    for n in tree_.allNodes():
        if n.isTerminal():
            n.root = "terminalNode"
    s = tree_.__str__().replace("(", " ( ").replace(")", " ) ").replace('terminalNode', "")
    s = s.split()
    s.append("#ENDOFINPUT#")
    return s

def oneK(dataset):
    n = len(dataset)
    v = numpy.zeros((n, n), dtype=int)
    for index, element in enumerate(dataset):
        v[index, dataset[element]] = 1
    return v

def createDataset(treeList, dtk_generator, symbols):
    for string in treeList:
        t = tree.Tree(string)
        v = dtk_generator.dt(t)
        # partial parsing
        tokens = tokenizeTree(string)
        seq = [symbols[token] for token in tokens]

    return seq


#DTK Parameters

dimension = 1024
LAMBDA = 0.6
dtk_generator = dtk.DT(dimension=dimension, LAMBDA=LAMBDA)

Grammar = pickle.load(open("fullGrammarNormalized.txt", "rb"))      #full grammar
pathToTreeBank = "/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB3/"
# treesTraining = list(loadPennTree(pathToTreeBank, "23"))
# treesTesting = list(loadPennTree(pathToTreeBank, "24"))

string = "(S (INTJ no) (, ,) (NP it) (VP (VBD was) (RB n't) (NP (NNP black) (NNP monday))) (. .))"


symbols = dict(Grammar.symbols, **{"#ENDOFINPUT#": 80, "(": 81, ")": 82})
symbolsSize = len(symbols)

print (createDataset([string], dtk_generator, symbols))









# KERAS
model = Sequential()
# let's encode this vector sequence into a single vector
model.add(GRU(input_dim = dimension, output_dim=symbolsSize, return_sequences=False))
# model.add(Dense(input_dim = symbolsSize, output_dim=symbolsSize))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# "images" is a numpy float array of shape (nb_samples, nb_channels=3, width, height).
# "captions" is a numpy integer array of shape (nb_samples, max_caption_len)
# containing word index sequences representing partial captions.
# "next_words" is a numpy float array of shape (nb_samples, vocab_size)
# containing a categorical encoding (0s and 1s) of the next word in the corresponding
# partial caption.


model.fit()
