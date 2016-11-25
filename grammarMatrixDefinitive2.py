import numpy
from numpy import tanh
import sys
import random

from itertools import islice

sys.path.append("/home/ferrone/pyDTK_sync2/src")

import pickle
from tree import Tree as tree
from grammar import Rule, Grammar

from loadPennTree import loadPennTree

from grammarMatrixDefinitive import symbolList, posList, ruleList, symbolMatrix, fromRuleToSymbols, encodePos, W_matrix, hs

numpy.set_printoptions(precision=2)

def findElementByIndex(elementList, indexList):
    indexes = numpy.argwhere(indexList > 0.5)
    indexes = [x[0] for x in indexes]
    return numpy.array(elementList)[indexes]

def predict(sentence, poss, rules, syms):
    predictions = []

    P = encodePos(pos, 'vector')

    p = len(poss)
    r = len(rules)
    s = len(syms)

    S = symbolMatrix(poss, rules)

    W = W_matrix(poss, rules)

    sym = numpy.zeros(s)
    h = numpy.zeros(p + r)

    for token in sentence:
        # h1 = h[:p]
        # h2 = h[p:]
        # h2_ = blockNormalize(h2, groupRulesByRoots(rules))
        # h = numpy.concatenate((h1, h2_))

        v = W.dot(P[token])
        h = hs(S.dot(sym) + v)   # h è il vettore di REGOLE

        # print ('.........')
        # print ('pos:', findElementByIndex(pos, v))
        # print ('rules:', findElementByIndex(rules, h[p:]))

        predictions.append(h)

        #poi trasformo le regole in simboli
        for i, element in enumerate(h[p:]):
            if element > 0.2:
                rule = rules[i]
                root = rule.left
                s = numpy.argwhere(numpy.array(syms) == root)
                sym[s] = 1.
        sym_ = sym[p:]
        sym = numpy.concatenate((h[:p], sym_))
        # sym = fromRuleToSymbols(h, poss, rules, syms)
        indexSym = numpy.argwhere(sym > 0)
        indexSym = [x[0] for x in indexSym]
        # print ('sym', numpy.array(syms)[indexSym])



    extrasteps = 0
    for extra in range(extrasteps):
        h = hs(S.dot(sym))
        for i, element in enumerate(h[p:]):
            if element > 0.2:
                rule = rules[i]
                root = rule.left
                s = numpy.argwhere(numpy.array(syms) == root)
                sym[s] = 1.
        sym_ = sym[p:]
        sym = numpy.concatenate((h[:p], sym_))
    return predictions



if __name__ == '__main__':

    PennTreePath = "/home/ferrone/Datasets/PTB2/"

    sections_train = "02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21".split()

    treeList_train = loadPennTree(PennTreePath, sections_train, normalize=True)
    treeList_test = loadPennTree(PennTreePath, ['23'], normalize=True)
    treeList_valid = loadPennTree(PennTreePath, ['24'], normalize=True)

    Grammar = pickle.load(open('binaryGrammarMostFrequent1500.txt', 'rb'))

    pos = posList(Grammar)
    rules = ruleList(Grammar)
    syms = symbolList(pos, rules)

    pr = pos + rules

    S = symbolMatrix(pos, rules)




    sentence = ['DT', 'NP', 'NNP', 'VB']

    print('parsing:', sentence)
    predictions = predict(sentence, pos, rules, syms)
    #
    #
    #

    # for p in predictions:
    #     indexes = numpy.argwhere(p > 0)
    #     indexes = [i[0] for i in indexes]
    #     print (numpy.array(pr)[indexes])


    # sys.exit(0)


    for t in islice(treeList_train, 10):
        sentence = [x[0] for x in t.sentence_(True)]
        print (t)
        print (sentence)

        print ('parsing')

        predictions = predict(sentence, pos, rules, syms)

        # for p in predictions:
        #     indexes = numpy.argwhere(p > 0)
        #     indexes = [i[0] for i in indexes]
        #     print (numpy.array(pr)[indexes])


        prediction = predictions[-1]
        prediction = findElementByIndex(rules, prediction[len(pos):])
        # analyze rules not found
        rules_original = list(set(Rule.fromTree(r) for r in t.allRules()))
        rules_original = [r for r in rules_original if not r.terminalRule]

        found = []
        not_found = []
        for r in rules_original:
            if r.terminalRule:
                continue
            if r in prediction:
                found.append(r)
            else:
                not_found.append(r)

        for nf in not_found:
            print('not found', nf, nf in rules)

        print ('original:', len(rules_original))

        print ('recall: ', len(found)/len(rules_original))
        print ('precision: ', len(found)/len(prediction))
        print (len(prediction))

        print('...')



    sys.exit(0)
