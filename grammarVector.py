__author__ = 'lorenzo'


import numpy
import sys
import random

from itertools import islice

if sys.platform == 'darwin': # sto sul mio pc
    sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
else: #sto deployando su venere
    sys.path.append("/home/ferrone/pyDTK_sync2/src")

import pickle
from tree import Tree as tree
import grammar
import grammarMatrix as gm

from loadPennTree import loadPennTree

from keras.preprocessing.sequence import pad_sequences

def ruleVector(tree, grammar_):
    v = numpy.zeros(len(grammar_))
    x = [grammar.Rule.fromTree(x) for x in tree.allRules() if not x.terminalRule]
    for r in x:
        if r in grammar_:
            v[grammar_.index(r)] = 1
    return v

def posIndex(seq, pos):
    ss = [x[0] for x in seq.sentence_(True)]
    pos_indexes = []
    for p in ss:
        if p in pos:
            pos_indexes.append(pos.index(p))
        else:
            pos_indexes.append(0)

    return pos_indexes

def posVector(posIndex, k):
    l = []
    for n in posIndex:
        v = numpy.zeros(k)
        v[n] = 1
        l.append(v)
    return l

def ornateMatrix(M, k):
    # return a matrix of the form
    # (O | I)
    # (M | O)
    #  where I is a kxk identity matrix
    n = len(M)
    I = numpy.eye(n + k, k)
    Z = numpy.zeros((k, n))
    m = numpy.vstack((Z, M))
    return numpy.hstack((m, I))

def completeMatrix(Mw, Mu):
    r, c = Mw.shape
    a = numpy.hstack((Mw, Mu))
    I = numpy.eye(c - r)
    Z = numpy.zeros((c - r, c))
    b = numpy.hstack((Z, I))
    return numpy.vstack((a, b))

def predictWithMatrix(sentence, matrix):
    z = numpy.zeros((matrix.shape[0]))
    result = numpy.hstack((z, sentence[0]))
    for v in sentence[1:]:
        result = numpy.dot(matrix, result)
        result = numpy.hstack((result, v))

    return result

def terminalRuleList(grammar):
    nt_rules  = []

    for r in grammar.terminalrules.items():
        nt_rules.extend(r[1])
    nt_rules = list(set(nt_rules))
    return sorted(nt_rules)

def posList(grammar):
    return sorted(list(set([x.left for x in terminalRuleList(grammar)])))

if __name__ == '__main__':
    PennTreePath = "/home/ferrone/Datasets/PTB2/"
    # Grammar = pickle.load(open("binaryGrammar23.txt", "rb"))

    # creating new grammar ordered by frequency
    # sections = "00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22".split()
    sections_train = "02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21".split()

    treeList_train = loadPennTree(PennTreePath, sections_train, normalize=True)
    treeList_test = loadPennTree(PennTreePath, ['23'], normalize=True)
    treeList_valid = loadPennTree(PennTreePath, ['24'], normalize=True)

    #Grammar = grammar.Grammar.fromTrees(treeList, 15)
    #pickle.dump(Grammar, open('binaryGrammarMostFrequent15.txt', 'wb'))

    Grammar = pickle.load(open('binaryGrammarMostFrequent1500.txt', 'rb'))
    pos = posList(Grammar)
    rules = gm.listOfrules(Grammar)

    print (len(pos))
    print (len(rules))

    rules = pos + rules



    # rs = ['A', 'B', 'C',
    #       grammar.Rule('X', ['A', 'B']),
    #       grammar.Rule('Y', ['B', 'C']),
    #       grammar.Rule('Z', ['A', 'C']),
    #       grammar.Rule('T', ['X', 'Y']),
    #       grammar.Rule('U', ['Y', 'Z']),]
    #
    # print (rs)
    #
    # M = gm.grammarMatrix(rs)
    # print ('grammar: ')
    # print (M.T)
    #
    # T = gm.terminalRuleMatrix(rs)
    # print ('pos: ')
    # print (T.T)
    #
    # sys.exit(0)

    # pos = gm.fromPosListToRule(Grammar.posTags)
    # rules = pos + rules



    sys.exit(0)

    M = gm.grammarMatrix(rules)
    numpy.save('grammarMatrix_prova_2', M.T)
    # M = numpy.load('grammarMatrix_prova.npy')

    T = gm.terminalRuleMatrix(rules)
    numpy.save('posMatrix_prova_2', T.T)
    # T = numpy.load('posMatrix_prova.npy')
    print (M.shape)
    print (T.shape)
