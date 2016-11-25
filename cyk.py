__author__ = 'lorenzo'

import numpy
import sys
import itertools
import os
import psutil
import random
import pickle
import gc
# sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
#sys.path.append("/home/ferrone/pyDTK2/src")

from pydtk import dtk
from pydtk import operation as op
from pydtk.tree import Tree as tree

import grammar as gramm
import treeToCYKMatrix



class CYK:
    def __init__(self, dimension, LAMBDA, grammar, filter=2, operation=op.fast_shuffled_convolution):
        self.dimension = dimension
        self.LAMBDA = LAMBDA
        self.dtk_generator = dtk.DT(dimension=dimension, LAMBDA=LAMBDA, operation=operation)
        self.rule_cache = {}
        self.grammar = grammar
        self.filter = filter
        self.operation = operation

    def cleanCache(self):
        self.dtk_generator.cleanCache()
        self.rule_cache = {}
        gc.collect()

    def sorting_method(self, t1, second_object):
        sn = self.dtk_generator.sn(t1)
        return numpy.dot(sn, second_object)/numpy.dot(sn, sn)

    def filterRule(self, rule, distributed_vector, filter):
        if rule in self.rule_cache:
            return self.rule_cache[rule]
        else:
            ruleTree = rule.toTree()
            numNodes = len(list(ruleTree.allNodes()))
            ruleDTF = self.dtk_generator.dtf(ruleTree)
            score = numpy.dot(ruleDTF,distributed_vector)
            norm = numpy.dot(self.dtk_generator.dtf(ruleTree), self.dtk_generator.dtf(ruleTree))
            score = score/norm
            self.rule_cache[rule] = (score > numpy.power(self.LAMBDA, numNodes/2)/filter, score)
            return self.rule_cache[rule]

    def parse(self, sentence, k_best, distributed_vector=None, referenceTable=None):

        words = sentence.split()
        n = len(words)

        #initialize TABLE
        P = numpy.zeros((n, n), dtype=object)
        for i, _ in numpy.ndenumerate(P):
            P[i] = []

        #unit production
        for i, word in enumerate(words):
            # to prevent uncovered words we create rule of the form X -> w
            # for each symbol X in the grammar and for each word w in the sentence
            for symbol in self.grammar.symbols:
                rule = gramm.Rule(symbol,[word])    # create a new rule
                rt = rule.toTree()                  # and transform into tree

                score = numpy.dot(self.dtk_generator.sn(rt), distributed_vector)
                ## NORMALIZATION
                score = score/numpy.sqrt(numpy.dot(self.dtk_generator.sn(rt), self.dtk_generator.sn(rt)))
                rt.score = score

                P[i][0].append(rt)

            P[i, 0] = sorted(P[i, 0], key = lambda x: x.score, reverse=True)[:2]

        #non terminal rules
        numero_dtk = 0 #count iterations for debugging purpose
        for i in range(2, n + 1):
            #TODO:
            #add a check if numero_dtk is too high and break returning "not parsed"
            # total_size = len(dtk_generator.dt_cache) + len(dtk_generator.sn_cache) + len(dtk_generator.dtf_cache)
            # total_size_mbytes = (total_size*8*dtk_generator.dimension)/1048576
            # print (i, total_size_mbytes)
            if psutil.virtual_memory().percent > 95:
                return False, None, P

            for j in range(1, n - i + 2):
                for k in range(1, i):
                    # look for combination of a tree in leftCell with a tree in rightCell
                    leftCell = P[j - 1, k - 1]
                    rightCell = P[j + k - 1, i - k - 1]

                    for (subtree1, subtree2) in itertools.product(leftCell, rightCell):
                        stringa = subtree1.root + " " + subtree2.root
                        for rule in self.grammar.nonterminalrules[stringa]:
                            #FILTER on rules with too low score
                            passed, ruleScore = self.filterRule(rule, distributed_vector, self.filter)
                            if passed:
                                rtt = tree(root=rule.left, children=[subtree1, subtree2])
                                score = numpy.dot(self.dtk_generator.sn(rtt), distributed_vector)
                                ## NORMALIZATION
                                score = score/ruleScore
                                rtt.score = score

                                P[j-1, i-1].append(rtt)

                                numero_dtk = numero_dtk + 1

                #sort rules
                #P[j-1][i-1] = sorted(P[j-1][i-1], key=lambda x: x[1][1], reverse=True)
                P[j-1, i-1] = sorted(P[j-1, i-1], key=lambda x: x.score, reverse=True)
                #another k_best rules where the root is different than the first rule selected before
                #lista_diversi = [x for x in P[j-1][i-1] if x[0][0].left != P[j-1][i-1][0][0][0].left][:k_best]

                lista_diversi = [x for x in P[j-1, i-1] if x.root != P[j-1, i-1][0].root][:k_best]

                P[j-1, i-1] = P[j-1, i-1][:k_best]
                #if the new rules weren't already selected, add them
                if lista_diversi:
                    for a in lista_diversi:
                        if a not in P[j-1, i-1]:
                            P[j-1, i-1].append(a)

        #list of tree in the final cell of the table
        finalList = P[0, -1]
        if finalList:
            #final sort (by DTK)
            finalList = sorted(finalList, key=lambda x: numpy.dot(self.dtk_generator.dt(x),distributed_vector), reverse=True)
            return True, finalList , P
        else:
            #treeToCYKMatrix.printCYKMatrix(simpleTable(P))
            return False, None, P


if __name__ == "__main__":

    gs = []
    for i in range(2300, 2301):
          gs.append(open("/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB2/23/wsj_" + str(i) + ".mrgbinarized.txt"))


    lista_alberi = []
    conta = 0
    for g in gs:
        for l in g.readlines():
            conta = conta + 1
            l.strip("\n")
            ts = tree(string=l)
            ts = ts.binarize()
            ts = ts.normalize()
            lista_alberi.append(ts)


    G = pickle.load(open("grammarPennTree5.txt", "rb"))

    distributed = dtk.DT(dimension=1024, LAMBDA=0.4, operation=op.fast_shuffled_convolution)


    cykInstance = CYK(1024, 0.6, G)

    for t in lista_alberi:
        sent = tree.sentence_(t)
        v = distributed.dt(t)
        isParsed, parseList, parseMatrix = cykInstance.parse(sent, 1, v)
