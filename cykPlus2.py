import numpy
import sys
import itertools
import os
import psutil
import pickle
import gc
import codecs
import random
import time
# sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
#sys.path.append("/home/ferrone/pyDTK2/src")

from pydtk import dtk
from pydtk import operation
from pydtk.tree import Tree
from pydtk.treekernel import TreeKernel as TK

import grammar as gramm
from grammar import Rule

import metrics
from treeToCYKMatrixPlus import treeToCYKMatrix, printCYKMatrix, comparator, partialRulesMatrix


def topRule(t):
    return Rule(t.root, [x.root for x in t.children])


class CYKPlus:
    def __init__(self, dimension, LAMBDA, grammar, filter=2, operation=operation.fast_shuffled_convolution):
        self.dimension = dimension
        self.LAMBDA = LAMBDA
        self.dtk_generator = dtk.DT(dimension=dimension, LAMBDA=LAMBDA, operation=operation)
        self.rule_cache = {}
        self.tree_cache = {}
        self.grammar = grammar
        self.filter = filter
        self.operation = operation

    def cleanCache(self):
        self.dtk_generator.cleanCache()
        self.rule_cache = {}
        self.tree_cache = {}
        gc.collect()

    def scorePartialRule(self, partialRule, filter, distributedVector = None):
        # gives a score to a list of tree of the form [t_1, ... t_n, "."], each of which has a score
        ruleString = " ".join(c.root for c in partialRule)      # A B
        scores = []
        for rule in self.grammar.nonterminalrules[ruleString]: # X -> A B C D
            # filtrare regole
            passed, score = self.filterRule(rule, distributedVector, self.filter)
            scores.append(score)
        if scores:
            # return max(scores) * numpy.mean([numpy.dot(self.dtk_generator.dtf(x), distributedVector) for x in partialRule])
            return max(scores) * numpy.mean([x.score/numpy.sqrt(len(list(x.allNodes()))) for x in partialRule])
        else:
            return 0

    def filterRule(self, rule, distributed_vector, filter):
        ruleTree = rule.toTree()
        passed, score =  self.filterTree(ruleTree, distributed_vector, filter)
        numNodes = len(list(ruleTree.allNodes()))
        score = score/numpy.power(self.LAMBDA, numNodes/2)
        if score > 1.2:
            return passed, 1.2
        else:
            return passed, score

            # numNodes = len(list(ruleTree.allNodes()))
            # ruleDTF = self.dtk_generator.dtf(ruleTree)
            # score = numpy.dot(ruleDTF,distributed_vector)
            # # norm = numpy.dot(self.dtk_generator.dtf(ruleTree), self.dtk_generator.dtf(ruleTree))
            # # score = score/norm
            # self.rule_cache[rule] = (score > numpy.power(self.LAMBDA, numNodes/2)/filter, score)
            # return self.rule_cache[rule]

    def filterTree(self, tree, distributed_vector, filter):
        if tree in self.tree_cache:
            return self.tree_cache[tree]
        else:
            numNodes = len(list(tree.allNodes()))
            if tree.depth() == 2: # is a rule
                try:
                    ruleDTF = tree.vector
                except AttributeError:
                    ruleDTF = self.dtk_generator.dtf(tree)
                    tree.vector = ruleDTF
            else:
                ruleDTF = self.dtk_generator.dtf(tree)

            score = numpy.dot(ruleDTF,distributed_vector)
            # norm = numpy.dot(self.dtk_generator.dtf(tree), self.dtk_generator.dtf(tree))
            # score = score/norm
            self.tree_cache[tree] = (score > numpy.power(self.LAMBDA, numNodes/2)/filter, score)
            return self.tree_cache[tree]


    def parse(self, sentence, k_best=2, distributed_vector=None, referenceTable=None, rule_filter=2, realTree = None):
        start = time.time()
        """return the k-best parse"""
        words = sentence.split()
        n = len(words)

        #initialize TABLE
        C = numpy.zeros((n, n), dtype=object)
        for i, _ in numpy.ndenumerate(C):
            #each cell has a type1 list and a type2 list (C is matrix of completed (up to that point) trees)
            #elements of type1 are complete trees: A -> B C D ... (B, C, D ... sono alberi completi)
            #elements of type2 are LIST of partial trees: [B, C, ..., .] (B, C ... sono ancora alberi completi, ma esiste una regola A -> B C D .... )
            #each element in C should also have a score attached to it (<dtk(element), dtk(reference_tree)> <- o qualche variazione sul tema )
            C[i] = [[], []]

        #unit production
        # start_unit = time.time()
        # total_time_symbols = 0
        # total_time_sort = 0
        for i, word in enumerate(words):
            # to prevent uncovered words we create rule of the form X -> w
            # for each symbol X in the grammar and for each word w in the sentence

            # TODO: also, do more clever stuff: i.e if w is a number always do CD -> w
            # TODO: and the same for punctuation

            # 1) parsing step

            # some special cases:
            if word == ",":
                tree = gramm.Rule(",", word)
                rt = tree.toTree()
                score = numpy.dot(self.dtk_generator.dtf(rt), distributed_vector)
                rt.score = score
                # in this cases I don't think I need to filter, because by definition we take the *right* choice
                # if score > self.LAMBDA/self.filter:
                C[i, 0][0].append(rt)

            elif word in "`'":
                tree = gramm.Rule(2*word, word)
                rt = tree.toTree()
                score = numpy.dot(self.dtk_generator.dtf(rt), distributed_vector)
                rt.score = score
                # in this cases I don't think I need to filter, because by definition we take the *right* choice
                # if score > self.LAMBDA/self.filter:
                C[i, 0][0].append(rt)

            else:
                for symbol in self.grammar.posTags:     # prendere lista solo dei POS
                    tree = gramm.Rule(symbol,[word])    # create a new rule
                    rt = tree.toTree()                  # and transform into tree

                    #compute and normalize score
                    score = numpy.dot(self.dtk_generator.dtf(rt), distributed_vector)
                    # score = score/numpy.sqrt(numpy.dot(self.dtk_generator.sn(rt), self.dtk_generator.sn(rt))) #prova senza normalizzazione
                    rt.score = score
                    if score > self.LAMBDA/self.filter:
                        C[i, 0][0].append(rt)
            # total_time_symbols = total_time_symbols + (time.time() - start_unit_symbols)

            C[i, 0][0] = sorted(C[i, 0][0], key = lambda x: x.score, reverse=True)[:k_best] # prima era [:k_best], a volte la prima scelta è sbagliata...

            # 2) self-filling step
            for tree in C[i, 0][0]:  #rule = A -> w
                treeString = tree.root
                rules = self.grammar.nonterminalrules[treeString] #X -> A .

                incompleteRules = False
                completeRules = []
                for rule in rules:
                    if treeString != " ".join(rule.right):
                        incompleteRules = True
                    else:
                        completeRules.append(rule)
                # incompleteRules = [rule for rule in rules if treeString != " ".join(rule.right)]
                # completeRules = [rule for rule in rules if treeString == " ".join(rule.right)]

                if incompleteRules:
                    C[i, 0][1].append([tree])

                # for incompleteRule in incompleteRules:
                #     passed, score = self.filterRule(incompleteRule, distributed_vector, self.filter)
                #     if passed:
                #         C[i, 0][1].append([tree])
                #         break

                for completeRule in completeRules:
                    passed, score = self.filterRule(completeRule, distributed_vector, self.filter)
                    if passed:
                        # it's a complete rule (of the form X -> A )
                        newTree = Tree(root=completeRule.left, children=[tree])
                        newTreescore = numpy.dot(self.dtk_generator.sn(newTree), distributed_vector)
                        passed, score = self.filterTree(newTree, distributed_vector, self.filter)

                        if passed: #pensare ad un filtro più stringente....
                            newTree.score = newTreescore
                            #print (new_tree)
                            C[i, 0][0].append(newTree)

                            if len(C[i, 0][0]) > 100:
                                print ('aiuto')
                                break

                # for rule in rules: #rule X -> A B C
                #     passed, score = self.filterRule(rule, distributed_vector, self.filter)
                #     if passed:
                #         # ulteriore filtro, se la regola ha un punteggio "alto", non provare ad espanderla ancora...?
                #         if treeString != " ".join(rule.right):
                #             # it's a partial rule
                #             if [tree] not in C[i, 0][1]:
                #                 C[i, 0][1].append([tree])
                #         else:
                #             # it's a complete rule (of the form X -> A )
                #             newTree = Tree(root=rule.left, children=[tree])
                #             newTreescore = numpy.dot(self.dtk_generator.sn(newTree), distributed_vector)
                #             passed, score = self.filterTree(newTree, distributed_vector, self.filter)
                #
                #             if passed: #pensare ad un filtro più stringente....
                #                 newTree.score = newTreescore
                #                 #print (new_tree)
                #                 C[i, 0][0].append(newTree)


            #sort and trimming
            if len(C[i, 0][0]) > k_best:
                # print (len(C[i, 0][0]))
                C[i,0][0] = sorted(C[i,0][0], key=lambda x: x.score, reverse=True)[:k_best] #[:k_best]
            # start_sort = time.time()
            # print (len(C[i, 0][1]))

            #C[i,0][1] = sorted(C[i,0][1], key=lambda x: self.scorePartialRule(x, self.filter, distributed_vector), reverse=True)[:k_best]
            # total_time_sort = total_time_sort + (time.time() - start_sort)

        #unit production finished, printing for debug
        # for i, word in enumerate(words):
        #     print (word)
        #     for p in C[i, 0][0]:
        #         print (p)
        #         print ("-")
        #     print ("--")

        # print ('fine unit production', time.time() - start_unit)
        # print ('fine symbol production', total_time_symbols)
        # print ('sorting time', total_time_sort)

        start_unit = time.time()
        # after unit rules
        for i in range(2, n + 1):
            for j in range(1, n - i + 2):

                # 1) parsing
                for k in range(1, i):
                    # look for combination of a tree in leftCell with a tree in rightCell
                    leftCell = C[j - 1, k - 1]
                    rightCell = C[j + k - 1, i - k - 1]
                    for (partialRule, completeRule) in itertools.product(leftCell[1], rightCell[0]):
                        ruleString = " ".join(c.root for c in partialRule) + " " + completeRule.root
                        rules = self.grammar.nonterminalrules[ruleString]

                        # provare a dividere in regole complete e parziali e filtrare/ordinare dopo
                        newPartialRule = False
                        newCompleteRule = []
                        for rule in rules:
                            if " ".join(rule.right) == ruleString:
                                newCompleteRule.append(rule)
                            else:
                                newPartialRule = True

                        children = partialRule + [completeRule]

                        if newPartialRule:
                            C[j-1, i-1][1].append(children)

                        for rule in newCompleteRule:
                            passed, ruleScore = self.filterRule(rule, distributed_vector, self.filter)
                            if rule == Rule(left = "NP", right=["NP , NP , NP , NP , NP , NP CC NP"]):
                                t = rule.toTree()
                                v = numpy.linalg.norm(self.dtk_generator.dtf(t))
                                print (v)


                            # if passed != (rule in [gramm.Rule.fromTree(x) for x in realTree.allRules()]):
                            #     print (i, j, rule, ruleScore, passed, rule in [gramm.Rule.fromTree(x) for x in realTree.allRules()])
                            if passed:
                                # print (i, j, rule, ruleScore, rule in [gramm.Rule.fromTree(x) for x in realTree.allRules()])

                                newTree = Tree(root=rule.left, children=children)
                                score = numpy.dot(self.dtk_generator.sn(newTree), distributed_vector)
                                newTree.score = score
                                if newTree not in C[j - 1, i-1][0]:
                                    C[j-1, i-1][0].append(newTree)

                # 2) self-filling
                for tree in C[j-1, i-1][0]:
                    ruleString = tree.root
                    rules = self.grammar.nonterminalrules[ruleString]

                    incompleteRules = False
                    completeRules = []
                    for rule in rules:
                        if ruleString != " ".join(rule.right):
                            incompleteRules = True
                        else:
                            completeRules.append(rule)

                    if incompleteRules:
                        C[j-1, i-1][1].append([tree])

                    for completeRule in completeRules:
                        # filter on rule with low score
                        passed, ruleScore = self.filterRule(completeRule, distributed_vector, self.filter)
                        if passed:
                            # TODO: add a check to prevent chain longer than X -> X
                            if (len(tree.children) == 1) and (completeRule.left == tree.root == tree.children[0].root):
                                continue
                            newTree = Tree(root=completeRule.left, children=[tree])
                            score = numpy.dot(self.dtk_generator.sn(newTree), distributed_vector)
                            # passed, score2 = self.filterTree(newTree, distributed_vector, self.filter)

                            newTree.score = score

                            C[j-1, i-1][0].append(newTree)
                    if len(C[j-1][i-1][0]) > 50:
                        break
                                    #print ("dopo: ", len(C[i, j][0]), r)

                # stampa numero di nodi
                # for t in C[j-1, i-1][0]:
                #
                #     print (len(list(t.allNodes())), end=" ")
                # if C[j-1, i-1][0]:
                #     print ("numero nodi")

                # 3) sorting and trimming
                if len(C[j-1, i-1][0]) > k_best:
                    C[j-1, i-1][0] = sorted(C[j-1, i-1][0], key=lambda x: x.score, reverse=True)

                # if C[j-1][i-1][0]:
                #     print (i, j, C[j-1][i-1][0])

                # as in cyk normale, add a list of "different" rules
                lista_diversi = [x for x in C[j-1, i-1][0] if x.root != C[j-1, i-1][0][0].root][:k_best]

                #e solo dopo trimmare a k_best
                C[j-1,i-1][0] = C[j-1,i-1][0][:k_best]
                #if the new rules weren't already selected, add them
                if lista_diversi:
                    for a in lista_diversi:
                        if a not in C[j-1, i-1][0]:
                            C[j-1, i-1][0].append(a)

                #infine sorto e trimmo l'altra lista
                start_sort = time.time()
                if len(C[j-1, i-1][1]) > k_best:
                    # print ("numero di regole parziali: ", len(C[j-1, i-1][1]))
                    # for pr in C[j-1, i-1][1]:
                    #     for t in pr:
                    #         print (t.root, end= " ")
                    #     print (" - ", end = " ")
                    # print()
                    # if (j-1, i-1) == (0, 24):
                    #     l = sorted(C[j-1, i-1][1], key=lambda x: self.scorePartialRule(x, self.filter, distributed_vector), reverse=True)
                    #     print ("cella 0 24: ", [([x.root for x in t], self.scorePartialRule(t, self.filter, distributed_vector)) for t in l])

                    C[j-1, i-1][1] = sorted(C[j-1, i-1][1], key=lambda x: self.scorePartialRule(x, self.filter, distributed_vector), reverse=True)[:k_best]
                # total_time_sort = total_time_sort + (time.time() - start_sort)

        #rendo l'ouput come quello di CYK
        # print ('fine parsing', time.time() - start_unit)
        # print ('sorting time', total_time_sort)

        finalList = C[0][-1][0]

        # print ("time: ", time.time() - start)
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
    grammar = pickle.load(open("fullGrammarNormalized.txt", "rb"))      #full grammar
    print ("grammar loaded")

    # print (grammar.nonterminalrules["S , VP NP ."])





    treeString12 = "(S (PRP i) (VP (V saw)(NP (DET the)(N man))) (PP (P with)(NP (DET the)(N telescope))))"

    t = Tree(string = treeString12)
    t.binarize()
    t.normalize()

    grammar = gramm.Grammar.fromTrees([t])

    print (grammar.nonterminalrules)
    print (grammar.terminalrules)
    print (grammar.posTags)

    plusInstance = CYKPlus(dimension, l, grammar)


    # print (grammar.nonterminalrules["NP VP PP"])
    # print (grammar.terminalrules["man"])

    covered, regole = grammar.checkCoverage(t)
    print (covered, regole)

    cyk_matrix = treeToCYKMatrix(t)
    printCYKMatrix(cyk_matrix)

    print ("PARTIAL--")

    pr = partialRulesMatrix(t.children)
    printCYKMatrix(pr)

    sentence = Tree.sentence_(t)
    vector = distributed.dt(t)


    # cykM = treeToCYKMatrix.treeToCYKMatrix(t)
    # treeToCYKMatrix.printCYKMatrix(cykM)

    # for r in t.allRules():
    #     print (gramm.Rule.fromTree(r))

    #call the parser
    print ("parsing: ", sentence)
    print (t)

    #print (grammar.checkCoverage(t))
    isParsed, parseList, C = plusInstance.parse(sentence, distributed_vector=vector, k_best=2, rule_filter=2, realTree = t)






    #printing final matrix
    print ("====")
    # n, _ = C.shape

    for row in C:
        for x in row:
            print ([(topRule(t), t.score) for t in x[0]], end="\t")
        print ()


    for row in C:
        for x in row:
            print ([[x.root for x in t] for t in x[1]], end="\t")
        print ()


    if isParsed:
        print (parseList[0])
        print (parseList[0] == t)
    else:
        print ("notParsed")


    comparator(t, C)
