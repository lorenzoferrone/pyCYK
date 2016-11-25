import numpy
from numpy import tanh
import sys
import random

from itertools import islice

if sys.platform == 'darwin': # sto sul mio pc
    sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
else: #sto deployando su venere
    sys.path.append("/home/ferrone/pyDTK_sync2/src")

import pickle
from tree import Tree as tree
from grammar import Rule, Grammar


from loadPennTree import loadPennTree

numpy.set_printoptions(precision=2)

def listDifference(l1, l2):
    return sorted(list(set(l2) - set(l1)))

def posList(grammar):
    return sorted(list(set([x.left for x in terminalRuleList(grammar)])))

def terminalRuleList(grammar):
    nt_rules  = []
    for r in grammar.terminalrules.items():
        nt_rules.extend(r[1])
    nt_rules = list(set(nt_rules))
    return sorted(nt_rules)

def ruleList(Grammar):
    nt_rules  = []

    for r in Grammar.nonterminalrules.items():
        nt_rules.extend(r[1])
    nt_rules = list(set(nt_rules))
    return sorted(nt_rules)

def symbolList(poss, rules):
    syms = []
    for rule in rules:
        root, tail = rule.left, rule.right
        syms.extend([root, tail[0], tail[1]])
    syms = sorted(list(set(syms)))
    syms = poss + syms
    return syms #sorted(list(set(syms)))

def symbolMatrix(poss, rules):
    syms = symbolList(poss, rules)
    p = len(poss)
    r = len(rules)
    s = len(syms)
    S = numpy.zeros((p + r, s))
    for i in range(p):
        S[i, i] = 1
    for i, rule in enumerate(rules, p):
        a, b = rule.right
        if a in syms:
            ia = syms.index(a)
            ia = numpy.argwhere(numpy.array(syms[:]) == a)
            ia = [x[0] for x in ia]
            if len(ia) == 1:
                S[i, ia] = 0.4
            else:
                S[i, ia] = 0.2
        if b in syms:
            ib = syms.index(b)
            ib = numpy.argwhere(numpy.array(syms[:]) == b)
            ib = [x[0] for x in ib]
            if len(ib) == 1:
                S[i, ib] = 0.4
            else:
                S[i, ib] = 0.2

    return S

def fromRuleToSymbols(vector, poss, rules, syms):
    p = len(poss)
    indexes = numpy.argwhere(vector > 0)
    indexes = [x[0] for x in indexes]
    # print ('rules', numpy.array(poss + rules)[indexes])
    groups = groupBySymbol(vector[p:], syms[p:], rules)

    # print (groups, len(groups))


    # groups = divideByBlock(vector, groupIndexes)
    # print (vector, groupIndexes)
    v1 = vector[:p]
    # v2 = vector[p:]
    v2 = numpy.zeros(len(syms) - p)
    for i, group in enumerate(groups):
        g = hs(sum(group))
        v2[i] = g
    v = numpy.concatenate((v1, v2))
    # print (vector[26])
    # print (v)
    # print (v2)
    # indexSym = numpy.argwhere(v > 0)
    # print ('sym', numpy.array(syms)[indexSym])
    return v

def groupBySymbol(vector, symbols, rules):
    # qua dentro assumo che vector ha la stessa lunghezza di rules (non conto i pos)
    groups = []
    for symbol in symbols:
        g = [vector[i] for i, rule in enumerate(rules) if rule.left == symbol]
        groups.append(g)
    return groups

def groupBy(list):
    lastElement = None
    indexes = []
    for index, element in enumerate(list):
        if element != lastElement:
            indexes.append(index)
            lastElement = element
    return indexes[1:] + [len(list)]

def groupRulesByRoots(rules):
    lastRoot = None
    indexes = []
    for index, rule in enumerate(rules):
        root = rule.left
        if root != lastRoot:
            indexes.append(index)
            lastRoot = root
    return indexes[1:] + [len(rules)]

def blockNormalize(vector, blocks): #blocks = (n1, n2, n3, ...) indici FINALI dei blocchi
    l = divideByBlock(vector, blocks)
    r = [block/numpy.linalg.norm(block, 1) if numpy.linalg.norm(block, 1) != 0 else block for block in l ]
    return numpy.concatenate(r)

def divideByBlock(vector, blocks): #blocks = (n1, n2, n3, ...) indici FINALI dei blocchi
    blocks_ = [0] + blocks
    blocksRanges = [(blocks_[i], blocks_[i+1]) for i in range(len(blocks_)-1)]
    # print (blocksRanges)
    l = []
    for start, end in blocksRanges:
        l.append(vector[start:end])
    return l

def sigmoid(v):
    return 1 / (1 + numpy.exp(-v))

def gs(x):
    return 1 / (1 + numpy.exp(-10*(x - 0.5)))

def hs(v):
    return numpy.where(v >= 0.5, 1., 0.)

def relu(v):
    return numpy.minimum(v, 1)

def parseRule(string):
    left, right = string.split('->')
    left = left.strip()
    right = right.strip().split(" ")
    return Rule(left, right)

def oneHot(n, k):
    v = numpy.zeros(n)
    v[k] = 1
    return v

def encodePos(poss, mode='vector'):
    if mode == 'vector':
        return {pos: oneHot(len(poss), i) for i, pos in enumerate(poss)}
    else:
        return {pos: i for i, pos in enumerate(poss)}

def W_matrix(poss, rules):
    p = len(poss)
    r = len(rules)
    W = numpy.zeros((p + r, p))
    for i in range(p):
        W[i,i] = 1
    for i, rule in enumerate(rules, p):
        a, b = rule.right
        if a in poss:
            ia = poss.index(a)
            W[i, ia] = 0.4

        if b in poss:
            ib = poss.index(b)
            W[i, ib] = 0.4
    return W

def U_matrix(poss, rules):
    p = len(poss)
    r = len(rules)
    U = numpy.zeros((p + r, p + r))

    roots = numpy.array([r.left for r in rules])

    for i in range(p):
        U[i, i] = 1
    for i, rule in enumerate(rules, p):
        a, b = rule.right
        if a in poss:
            ia = poss.index(a)
            U[i, ia] = 0.4
        if b in poss:
            ib = poss.index(b)
            U[i, ib] = 0.4
        # check left side of rules
        # roots = numpy.array([r.left for r in rules])
        if a in roots:
            ia = p + numpy.where(roots == a)[0]
            U[i, ia] = 0.4 # /len(regole che hanno gli stessi figli)
        if b in roots:
            ib = p + numpy.where(roots == b)[0]
            U[i, ib] = 0.4
    return U

def encodeTree(t, posList):
    return [POS[x[0]] for x in t.sentence_(True)]

def predict(sentence, W, U):
    # sentence is list of poss
    pr, p = W.shape
    r = pr - p      #p = len(pos), r = len(rules)
    h = numpy.zeros(p + r)
    # print()
    predictions = []
    for token in sentence:
        h1 = h[:p]
        h2 = h[p:]
        h2_ = blockNormalize(h2, groupRulesByRoots(rules))
        h = numpy.concatenate((h1, h2_))
        v = W.dot(P[token])
        # h = hs(U.dot(h) + v)   # versione precedente
        h = U.dot(h)
        # h1 = h[:p]
        # h2 = h[p:]
        # h2_ = blockNormalize(h2, groupRulesByRoots(rules))
        # h = numpy.concatenate((h1, h2_))
        h = hs(h + v)


        # print (predictRules(h, pos, rules)[:], len(predictRules(h, pos, rules)))
        predictions.append(h)
        # print ('o\t', h)

    extrasteps = 0
    # for extra in range(extrasteps):
    #     h = numpy.concatenate((h[:80], blockNormalize(h[80:], groupRulesByRoots(rules))))
    #     h = hs(U.dot(h))
    return predictions

def predictRules(h, pos, rules):
    indexes = numpy.argwhere(h > 0.1)
    indexes = [i[0] for i in indexes]
    p = len(pos)
    pr = numpy.array(pos + rules)
    predict = []
    for i in indexes:
        if i < p:
            predict.append(pr[i])
            continue
        else:
            predict.append(pr[i])
    return predict


if __name__ == "__main__":

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
    rules = ruleList(Grammar)

    iRule = rules.index(parseRule('@ADJP -> @ADJP IN'))
    print (iRule)
    print (pos.index('@NP'))
    print (pos.index('IN'))
    print (pos.index('JJ'))

    W = W_matrix(pos, rules)
    U = U_matrix(pos, rules)


    syms = symbolList(pos, rules)
    print (syms)
    print (len(syms))

    # sys.exit(0)


    u187 = U[iRule + 80]
    index = [x[0] for x in numpy.argwhere(u187 > 0.1)]

    pr = pos + rules

    print ('line187', rules[iRule])
    for i in index:
        if i <= 80:
            print (pr[i])
        else:
            print (pr[i])


    # roots = set([r.left for r in rules])


    # sys.exit(0)

    P = encodePos(pos, 'vector')

    for t in islice(treeList_train, 1):
        sentence = [x[0] for x in t.sentence_(True)]
        print (t)
        print (sentence)

        print ('parsing')

        predictions = predict(sentence, W, U)
        lastPrediction = []
        for h in predictions:
            prediction = predictRules(h, pos, rules)

            newPreds = listDifference(lastPrediction, prediction[len(pos):])
            print (prediction[:len(pos)] + newPreds, len(prediction))
            lastPrediction = prediction[len(pos):]
            # print (prediction[:], len(prediction))


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
        print (numpy.linalg.norm(h, 1))

        print('...')
