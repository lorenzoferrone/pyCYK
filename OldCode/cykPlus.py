__author__ = 'lorenzo'

import numpy
import sys
import itertools
import os
import psutil

sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
#sys.path.append("/home/ferrone/pyDTK2/src")
import random
import dtk
import operation
import pickle
import gc
import codecs

import grammar as gramm

import metrics

from tree import Tree as tree
from treekernel import TreeKernel as TK
import treeToCYKMatrix

class CYKPlus:

    def __init__(self, dimension, LAMBDA, grammar, filter=2, operation=operation.fast_shuffled_convolution):
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

    def scorePartialRule(self, partialRule, distributedVector = None):
        # gives a score to a list of tree of the form [t_1, ... t_n, "•"], each of which has a score
        # tentativo 1: punteggio medio degli alberi nella lista....
        # tentativo 2: max degli alberi nella lista
        # tentativo 3: max su tutte le regole COMPLETE che la chiuderebbero

        # TODO: IMPORTANT: create a more reasonable version for this!!

        # 1
        # return sum(t.score for t in partialRule[:-1])/(len(partialRule) - 1)

        # 3
        ruleString = " ".join(c.root for c in partialRule[:-1])
        ruleScores = []
        for rule in self.grammar.nonterminalrules[ruleString]:
            ruleDTF = self.dtk_generator.dtf(rule.toTree())
            score = numpy.dot(ruleDTF, distributedVector)
            ruleScores.append(score)
        return max(ruleScores)




    def filterRule(self, rule, distributed_vector, filter):
        if rule in self.rule_cache:
            return self.rule_cache[rule]
        else:
            ruleTree = rule.toTree()
            numNodes = len(list(ruleTree.allNodes()))
            ruleDTF = self.dtk_generator.dtf(ruleTree)
            score = numpy.dot(ruleDTF,distributed_vector)
            norm = numpy.dot(self.dtk_generator.dtf(ruleTree), self.dtk_generator.dtf(ruleTree))
            punteggio_regola = score/norm
            self.rule_cache[rule] = (score > numpy.power(self.LAMBDA, numNodes/2)/filter, score)
            return self.rule_cache[rule]


    def parse(self, sentence, k_best=2, distributed_vector=None, referenceTable=None, rule_filter=2):
        """return the k-best parse"""
        words = sentence.split()
        n = len(words)

        #initialize TABLE
        C = numpy.zeros((n, n), dtype=object)
        for i, _ in numpy.ndenumerate(C):
            #each cell has a type1 list and a type2 list (C is matrix of completed (up to that point) trees)
            #elements of type1 are complete trees: A -> B C D ... (B, C, D ... sono alberi completi)
            #elements of type2 are LIST of partial trees: [B, C, ..., •] (B, C ... sono ancora alberi completi, ma esiste una regola A -> B C D .... )
            #each element in C should also have a score attached to it (<dtk(element), dtk(reference_tree)> <- o qualche variazione sul tema )
            C[i] = [[], []]

        #parsing step
        numero_dtk = 0
        for span in range(0, n):
            for i in range(0, n-span):
                j = i + span
                if i == j:
                    # to prevent uncovered words we create rule of the form X -> w
                    # for each symbol X in the grammar and for each word w in the sentence
                    for sym in self.grammar.symbols:
                        rule = gramm.Rule(sym,[words[i]])
                        rt = rule.toTree()

                        score = numpy.dot(self.dtk_generator.sn(rt), distributed_vector)
                        #score = numpy.dot(dtk_generator.dtf(rt), distributed_vector)
                        #score = sorting_method(dtk_generator, rt, distributed_vector)
                        ## NORMALIZATION
                        score = score/numpy.sqrt(numpy.dot(self.dtk_generator.sn(rt), self.dtk_generator.sn(rt)))
                        rt.score = score

                        C[i][j][0].append(rt)

                        #return None, []

                    C[i, j][0] = sorted(C[i, j][0], key=lambda x: x.score, reverse=True)[:k_best]

                    #self-filling part
                    #print ("prima: ", len(C[i, j][0]))
                    for B in C[i, j][0]:  #B = A -> B C
                        B_string = B.root
                        rules = self.grammar.nonterminalrules[B_string] #X -> A •

                        for r in rules:
                            if B_string != " ".join(r.right):
                                if [B, "•"] not in C[i, j][1]:    # <- devo dare uno score a questo (o forse no?)
                                    C[i, j][1].append([B, "•"])
                            else:
                                new_tree = tree(root=r.left, children=[B])
                                score = numpy.dot(self.dtk_generator.sn(new_tree), distributed_vector)
                                numero_dtk = numero_dtk + 1
                                #print (score, B.score, score > B.score)
                                if score > B.score: #pensare ad un filtro più stringente....
                                    new_tree.score = score
                                    #print (new_tree)
                                    C[i, j][0].append(new_tree)
                        if len(C[i, j][0]) > 10:
                            break
                    #print ("dopo: ", len(C[i][j][0]))

                    #sort and trimming (credo che non serva sortare l'altra lista...)
                    C[i,j][0] = sorted(C[i,j][0], key=lambda x: x.score, reverse=True)[:k_best]
                    #C[i,j][1] = sorted(C[i,j][1], key=lambda x: self.scorePartialRule(x, distributed_vector), reverse=True)[:k_best]


                if j > i:
                    for k in range(0, j):
                        first_cell_C = C[i, k]
                        second_cell_C = C[k + 1, j]
                        #print (len(first_cell_C[1]), len(second_cell_C[0]))
                        for (x,y) in itertools.product(first_cell_C[1], second_cell_C[0]):
                            xx = " ".join(c.root for c in x[:-1])
                            yy = y.root

                            string = xx + " " + yy

                            rules = self.grammar.nonterminalrules[string]
                            #print ("regole: ", len(rules), end=" ---- ")

                            for r in rules:
                                #rule filtering
                                passed, ruleScore = self.filterRule(r, distributed_vector, self.filter)
                                if passed:
                                    if " ".join(r.right) == string:
                                        #print (r, "empty")
                                        children = x[:-1]
                                        children.append(y)
                                        new_tree = tree(root=r.left, children=children)
                                        score = numpy.dot(self.dtk_generator.sn(new_tree), distributed_vector)
                                        numero_dtk = numero_dtk + 1
                                        new_tree.score = score
                                        if new_tree not in C[i, j][0]:
                                            C[i, j][0].append(new_tree)
                                    else:
                                        new_list = x[:-1] + [y] + ["•"]
                                        if new_list not in C[i, j][1]:
                                            C[i, j][1].append(new_list)

                    # TODO: devo vedere dove mettere il sorting... se qui, dopo il self-filling o in entrambi i posti. (o eventualmente con k diversi)
                    # TODO: sembra vada bene metterlo solo qui
                    # C[i, j][0] = sorted(C[i, j][0], key=lambda x: x.score, reverse=True)[:k_best]

                    #self-filling part
                    #print ("prima: ", len(C[i, j][0]))
                    for B in C[i, j][0]:
                        B_string = B.root #B = A -> B C
                        rules = self.grammar.nonterminalrules[B_string]
                        for r in rules:
                            #TODO: add another rule filter here?
                            passed, ruleScore = self.filterRule(r, distributed_vector, self.filter)
                            if passed:
                                if B_string != " ".join(r.right):
                                    if [B, "•"] not in C[i, j][1]:
                                        C[i, j][1].append([B, "•"])
                                else:
                                    # per evitare loop infiniti aggiungo un albero solo se il suo score è maggiore di quello precedente
                                    new_tree = tree(root=r.left, children=[B])
                                    score = numpy.dot(self.dtk_generator.sn(new_tree), distributed_vector)
                                    numero_dtk = numero_dtk + 1
                                    if score > B.score: #TODO: pensare ad un filtro più stringente (e che sicuro non crei loop infiniti) ??
                                        new_tree.score = score
                                        C[i, j][0].append(new_tree)
                                        #print ("dopo: ", len(C[i, j][0]), r)
                        if len(C[i, j][0]) > 20: # se ne sto aggiungendo troppi lascio perdere...
                            break

                    # sort (no trimming) la prima lista
                    C[i, j][0] = sorted(C[i, j][0], key=lambda x: x.score, reverse=True)

                    # as in cyk normale, add a list of "different" rules
                    lista_diversi = [x for x in C[i, j][0] if x.root != C[i, j][0][0].root][:k_best]

                    #e solo dopo trimmare a k_best
                    C[i, j][0] = C[i, j][0][:k_best]
                    #if the new rules weren't already selected, add them
                    if lista_diversi:
                        for a in lista_diversi:
                            if a not in C[j, i][0]:
                                C[i, j][0].append(a)

                    #infine sorto e trimmo l'altra lista
                    #C[i, j][1] = sorted(C[i, j][1], key=lambda x: self.scorePartialRule(x, distributed_vector), reverse=True)[:k_best]



        print (numero_dtk)
        #rendo l'ouput come quello di CYK_easy
        finalList = C[0][-1][0]
        if finalList:
            #final sort (by DTK)
            finalList = sorted(finalList, key=lambda x: numpy.dot(self.dtk_generator.dt(x),distributed_vector), reverse=True)
            return True, finalList , C
        else:
            #treeToCYKMatrix.printCYKMatrix(simpleTable(P))
            return False, None, C
        #return C, C[0][-1][0]



if __name__ == "__main__":

    dimension = 8192
    l = 0.6

    distributed = dtk.DT(dimension=dimension, LAMBDA=l, operation=operation.fast_shuffled_convolution)

    print ("loading grammar")
    grammar = pickle.load(open("fullGrammar.txt", "rb"))      #full grammar
    print ("grammar loaded")

    plusInstance = CYKPlus(dimension, l, grammar)

    #sentence listing
    treeString1 = "(S (CC but) (SBAR (IN while) (S (NP (DT the) (NNP new) (NNP york) (NNP stock) (NNP exchange)) (VP (VBD did) (RB n't) (VP (VB fall) (ADVP (RB apart)) (NP (NNP friday)) (SBAR (IN as) (S (NP (DT the) (NNP dow) (NNP jones) (NNP industrial) (NNP average)) (VP (VBD plunged) (NP (NP (CD 190.58) (NNS points)) (PRN (: --) (NP (NP (JJS most)) (PP (IN of) (NP (PRP it))) (PP (IN in) (NP (DT the) (JJ final) (NN hour)))) (: --)))))))))) (NP (PRP it)) (ADVP (RB barely)) (VP (VBD managed) (S (VP (TO to) (VP (VB stay) (NP (NP (DT this) (NN side)) (PP (IN of) (NP (NN chaos)))))))) (. .))"
    treeString2 = "(S (NP (NP (DT some) (`` ``) (NN circuit) (NNS breakers) ('' '')) (VP (VBN installed) (PP (IN after) (NP (DT the) (NNP october) (CD 1987) (NN crash))))) (VP (VBD failed) (NP (PRP$ their) (JJ first) (NN test)) (PRN (, ,) (S (NP (NNS traders)) (VP (VBP say))) (, ,)) (S (ADJP (JJ unable) (S (VP (TO to) (VP (VB cool) (NP (NP (DT the) (NN selling) (NN panic)) (PP (IN in) (NP (DT both) (NNS stocks) (CC and) (NNS futures)))))))))) (. .))"
    treeString3 = "(S (NP (NP (NP (DT the) (CD 49) (NN stock) (NN specialist) (NNS firms)) (PP (IN on) (NP (DT the) (NNP big) (NNP board) (NN floor)))) (: --) (NP (NP (DT the) (NNS buyers) (CC and) (NNS sellers)) (PP (IN of) (NP (JJ last) (NN resort))) (SBAR (WHNP (WP who)) (S (VP (VBD were) (VP (VBN criticized) (PP (IN after) (NP (DT the) (CD 1987) (NN crash)))))))) (: --)) (ADVP (RB once) (RB again)) (VP (MD could) (RB n't) (VP (VB handle) (NP (DT the) (NN selling) (NN pressure)))) (. .))"
    t = tree(string = treeString1)


    sentence = tree.sentence_(t)
    vector = distributed.dt(t)


    #call the parser
    print ("parsing: ", sentence)
    print (t)

    #print (grammar.checkCoverage(t))
    isParsed, parseList, _ = plusInstance.parse(sentence, distributed_vector=vector, k_best=3, rule_filter=2)

    print (_)

    if isParsed:
        print (parseList[0])
        print (parseList[0] == t)
    else:
        print ("notParsed")
