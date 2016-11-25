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

import grammar as gramm
from tree import Tree as tree
from treekernel import TreeKernel as TK
import treeToCYKMatrix


def sorting_method(dtk_generator, t1, second_object):
    sn = dtk_generator.sn(t1)
    return numpy.dot(sn, second_object)/numpy.dot(sn, sn)

def filterRule(rule, dtk_generator, distributed_vector, filter):
    if rule in rule_cache:
        return rule_cache[rule]
    else:
        ruleTree = rule.toTree()
        numNodes = len(list(ruleTree.allNodes()))
        ruleDTF = dtk_generator.dtf(ruleTree)
        score = numpy.dot(ruleDTF,distributed_vector)
        norm = numpy.dot(dtk_generator.dtf(ruleTree), dtk_generator.dtf(ruleTree))
        punteggio_regola = score/norm
        rule_cache[rule] = (score > numpy.power(dtk_generator.LAMBDA, numNodes/2)/filter, score)
        return rule_cache[rule]

rule_cache = {}
def parser_with_reconstruction3(sentence, grammar, k_best, distributed_vector=None, dtk_generator=None, referenceTable=None, rule_filter=2):
    #uso la grammatica nuova (grammar_2 )
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
        for symbol in grammar.symbols:
            rule = gramm.Rule(symbol,[word])    # create a new rule
            rt = rule.toTree()                  # and transform into tree

            score = numpy.dot(dtk_generator.sn(rt), distributed_vector)
            ## NORMALIZATION
            score = score/numpy.sqrt(numpy.dot(dtk_generator.sn(rt), dtk_generator.sn(rt)))
            rt.score = score

            #P[i][0].append(((rule, None),(rt, score)))
            P[i][0].append(rt)


        #P[i][0] = sorted(P[i][0], key=lambda x: x[1][1], reverse=True)[:2]
        P[i][0] = sorted(P[i][0], key = lambda x: x.score, reverse=True)[:2]

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
                leftCell = P[j - 1][k - 1]
                rightCell = P[j + k - 1][i - k - 1]

                for (subtree1, subtree2) in itertools.product(leftCell, rightCell):
                    stringa = subtree1.root + " " + subtree2.root
                    for rule in grammar.nonterminalrules[stringa]:
                        #FILTER on rules with too low score
                        passed, ruleScore = filterRule(rule, dtk_generator, distributed_vector, rule_filter)
                        if passed:
                            rtt = tree(root=rule.left, children=[subtree1, subtree2])
                            score = numpy.dot(dtk_generator.sn(rtt), distributed_vector)
                            ## NORMALIZATION
                            score = score/ruleScore
                            rtt.score = score

                            P[j-1][i-1].append(rtt)

                            numero_dtk = numero_dtk + 1

            #sort rules
            #P[j-1][i-1] = sorted(P[j-1][i-1], key=lambda x: x[1][1], reverse=True)
            P[j-1][i-1] = sorted(P[j-1][i-1], key=lambda x: x.score, reverse=True)
            #another k_best rules where the root is different than the first rule selected before
            #lista_diversi = [x for x in P[j-1][i-1] if x[0][0].left != P[j-1][i-1][0][0][0].left][:k_best]

            lista_diversi = [x for x in P[j-1][i-1] if x.root != P[j-1][i-1][0].root][:k_best]

            P[j-1][i-1] = P[j-1][i-1][:k_best]
            #if the new rules weren't already selected, add them
            if lista_diversi:
                for a in lista_diversi:
                    if a not in P[j-1][i-1]:
                        P[j-1][i-1].append(a)


            #PARTE DI DEBUG
            #se ho una reference, stampo la lista di regole che ho nella casella dopo aver trimmato e la casella corrispettiva
            #al primo errore ritorno Pp (stampata bene per confrontarla con referenceTable)

            if referenceTable is not None:
                if P[j-1][i-1] and referenceTable[i-1][j-1]:
                    lista_alberi = [x[0][0] for x in P[j-1][i-1]]
                    if referenceTable[i-1][j-1] not in lista_alberi:
                        #rule = P[j-1][i-1][0][0][0]

                        print ("cella: ", (i-1, j-1))

                        print ([x[0][0] for x in P[j-1][i-1]], referenceTable[i-1][j-1]) # <- questo caso è FAIL

                        #albero_sbagliato = P[j-1][i-1][0][1][0]
                        #score1 = P[j-1][i-1][0][1][1]
                        alberi_sbagliati = [x[1][0] for x in P[j-1][i-1]]



                        dtk_generator.dt_cache = {}
                        print ("SN: ")

                        for albero_sbagliato in alberi_sbagliati:

                            rtt = tree(root = referenceTable[i-1][j-1].left, children=alberi_sbagliati[0].children)

                            score1 = numpy.dot(dtk_generator.sn(albero_sbagliato), distributed_vector)
                            print (score1, albero_sbagliato)
                        score2 = numpy.dot(dtk_generator.sn(rtt), distributed_vector)
                        print (score2, rtt)

                        dtk_generator.dtf_cache = {}
                        print ("DTF: ")
                        for albero_sbagliato in alberi_sbagliati:
                            score1 = numpy.dot(dtk_generator.dtf(albero_sbagliato), distributed_vector)
                            regola = tree(root=albero_sbagliato.root, children=[tree(albero_sbagliato.children[0].root, None),tree(albero_sbagliato.children[1].root, None)])
                            print ("punteggio regola: ", numpy.dot(dtk_generator.dtf(regola), distributed_vector), regola)
                            print (score1, albero_sbagliato)
                        score2 = numpy.dot(dtk_generator.dtf(rtt), distributed_vector)
                        print (score2, rtt)
                        #return False, None, P
                else:
                    if referenceTable[i-1][j-1]: # e P[][] è vuota
                        pass
                        #print (P[j-1][i-1],referenceTable[i-1][j-1] ) # <- questo caso è FAIL
                        #return False, None, P
                    if P[j-1][i-1]: # e referenceTable è 0
                        pass
                        #print ("ok?", P[j-1][i-1],referenceTable[i-1][j-1] ) # <- questo caso può andar bene

            #FINE DEBUG

    #print (numero_dtk) #number of iteration

    #list of tree in the final cell of the table
    finalList = P[0][-1]
    if finalList:

        #final sort (by DTK)
        finalList = sorted(finalList, key=lambda x: numpy.dot(dtk_generator.dt(x),distributed_vector), reverse=True)
        return True, finalList , P
    else:
        #treeToCYKMatrix.printCYKMatrix(simpleTable(P))
        return False, None, P

def simpleTable(P):
    #new table
    n,_ = P.shape
    M = numpy.zeros((n, n), dtype=object)
    for (i,j), _ in numpy.ndenumerate(M):
        #print (P[i][0][0][0])
        if P[i,j]:
            if len ((P[i,j][0][0][0].right)) == 1:
                M[j, i] = [x[0][0].left for x in P[i, j]]
                #M[j,i] = P[i,j][0][0][0].left
            else:
                #M[j,i] = P[i,j][0][0][0]
                M[j,i] = [x[0][0] for x in P[i, j]]
        else:
            M[j,i] = 0

    return M


if __name__ == "__main__":

    #PARAMETER DEFINITION:
    #-grammar:

    #g = grammar(rules)
    #G = grammar.Grammar_(rules)






    distributed = dtk.DT(dimension=1024, LAMBDA=0.4, operation=operation.fast_shuffled_convolution)




    ss = "(S (@S (NP (@NP (@NP (NP (NNP Pierre)(NNP Vinken))(, ,))(ADJP (NP (CD 61)(NNS years))(JJ old)))(, ,))(VP (MD will)(VP (@VP (@VP (VB join)(NP (DT the)(NN board)))(PP (IN as)(NP (@NP (DT a)(JJ nonexecutive))(NN director))))(NP (NNP Nov.)(CD 29)))))(. .))"
    l = tree(string=ss)
    l = tree.binarize(l)
    l = tree.normalize(l)



    sent = tree.sentence_(l)
    print (l)
    print ()

    T = treeToCYKMatrix.treeToCYKMatrix(l)

    rules = [gramm.Rule.toTree(x) for x in l.allRules()]
    g = gramm.Grammar(rules)

    print (g.nonterminalrules)
    _, p, P = parser_with_reconstruction3(sent, g, 1, distributed.dt(l), distributed, referenceTable=T)


    print (p)
    
    

    #