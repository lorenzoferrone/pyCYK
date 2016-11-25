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


def simpleParser(sentence, grammar):
    """return True if the sentence is generated by the grammar (return the top node)"""
    #TODO: add backtracking (or reconstruction bottom-up)

    words = sentence.split()
    n = len(words)

    #initialize TABLE
    P = numpy.zeros((n, n), dtype=object)
    C = numpy.zeros((n, n), dtype=object)
    for i, _ in numpy.ndenumerate(P):
        #each cell has a type1 list and a type2 list (C is matrix of completed trees)
        P[i] = ([], [])
        C[i] = ([], [])

    #parsing step
    for span in range(0, n):
        for i in range(0, n-span):
            j = i + span
            if i == j:
                rules = grammar.terminalrules[words[i]]
                for r in rules:
                    if r.left not in P[i][j][0]:
                        P[i][j][0].append(r.left)

                #self-filling part
                for B in P[i, j][0]:
                    rules = grammar.nonterminalrules[B]
                    for r in rules:
                        if B != " ".join(r.right):
                            if B + "•" not in P[i, j][1]:
                                P[i, j][1].append(B + "•")
                        else:
                            if r.left not in P[i, j][0]:
                                P[i, j][0].append(r.left)
                                C[i][j][0].append(r.toTree())

            if j > i:
                for k in range(0, j):
                    first_cell = P[i, k]
                    second_cell = P[k + 1, j]

                    first_cell_C = C[i, k]
                    second_cell_C = C[k + 1, j]
                    #print ((i, j), (i, k), (k + 1, j))
                    #print (first_cell[1], second_cell[0])
                    for (x,y) in itertools.product(first_cell[1], second_cell[0]):
                        string = (x[:-1] + " " + y)
                        # print ("string: ", string)
                        rules = grammar.nonterminalrules[string]

                        for r in rules:

                            if " ".join(r.right) == string:
                                #print (r, "empty")
                                if r.left not in P[i, j][0]:
                                    P[i, j][0].append(r.left)
                                    C[i, j][0].append(r.toTree())
                            else:
                                lista_right = r.right
                                #prendo fino alla parte diversa
                                l = " ".join(r.right[:len(string.split())]) + "•"

                                #print (r, "not empty: ", l)
                                if l not in P[i, j][1]:
                                    P[i, j][1].append(l)


                #self-filling
                for B in P[i, j][0][:]:
                    rules = grammar.nonterminalrules[B]
                    #print ("SF: ", B, rules)
                    for r in rules:
                        #print ("-->", r)
                        if B != " ".join(r.right):
                            #print (B + "•" in P[i, j][1])
                            if B + "•" not in P[i, j][1]:
                                P[i, j][1].append(B + "•")

                        else:
                            if r.left not in P[i, j][0]:
                                P[i, j][0].append(r.left)
                                C[i, j][0].append(r.toTree())
            
    return P, P[0][-1][0]

def simpleParser2(sentence, grammar):
    """return True if the sentence is generated by the grammar (return the top node)"""
    #TODO: add backtracking (or reconstruction bottom-up)

    words = sentence.split()
    n = len(words)

    #initialize TABLE
    C = numpy.zeros((n, n), dtype=object)
    for i, _ in numpy.ndenumerate(C):
        #each cell has a type1 list and a type2 list (C is matrix of completed trees)
        C[i] = [[], []]

    #parsing step
    for span in range(0, n):
        for i in range(0, n-span):
            j = i + span
            if i == j:
                rules = grammar.terminalrules[words[i]]
                for r in rules:
                    if r.toTree() not in C[i][j][0]:
                        C[i][j][0].append(r.toTree())

                #self-filling part
                for B in C[i, j][0]:
                    B_string = B.root
                    rules = grammar.nonterminalrules[B_string]
                    #print (B_string, rules)
                    for r in rules:
                        if B_string != " ".join(r.right):
                            if [B, "•"] not in C[i, j][1]:
                                C[i, j][1].append([B, "•"])
                        else:
                            new_tree = tree(root=r.left, children=[B])
                            #print("new: ", new_tree)

                            if r.left not in [x.root for x in C[i, j][0]]:  # <- qua rischia il loop infinito.......
                                C[i][j][0].append(new_tree)


            if j > i:
                for k in range(0, j):
                    first_cell_C = C[i, k]
                    second_cell_C = C[k + 1, j]
                    #print ((i, j), (i, k), (k + 1, j))
                    #print (first_cell[1], second_cell[0])
                    #print (len(list(itertools.product(first_cell_C[1], second_cell_C[0]))))
                    for (x,y) in itertools.product(first_cell_C[1], second_cell_C[0]):
                        #print ("XY: ", x, "::",  y)
                        xx = " ".join(c.root for c in x[:-1])
                        yy = y.root
                        #print (" ".join(c.root for c in x.children[:-1]), y.root)

                        string = xx + " " + yy

                        rules = grammar.nonterminalrules[string]
                        if rules:
                            pass #print ("string: ", string, rules)
                        for r in rules:

                            if " ".join(r.right) == string:
                                #print (r, "empty")
                                children = x[:-1]
                                children.append(y)
                                new_tree = tree(root=r.left, children=children)
                                if new_tree not in C[i, j][0]:
                                    C[i, j][0].append(new_tree)
                            else:
                                new_list = x[:-1] + [y] + ["•"]
                                if new_list not in C[i, j][1]:
                                    C[i, j][1].append(new_list)

                #self-filling part
                for B in C[i, j][0]:
                    #print (i, j, B)
                    B_string = B.root
                    rules = grammar.nonterminalrules[B_string]
                    #print (B_string, rules)
                    for r in rules:
                        if B_string != " ".join(r.right):
                            if B not in C[i, j][1]:
                                C[i, j][1].append([B, "•"])
                        else:
                            new_tree = tree(root=r.left, children=[B])
                            #print("new: ", new_tree)
                            if r.left not in [x.root for x in C[i, j][0]]:  # <- pensare bene a sta cosa!!!
                                C[i][j][0].append(new_tree)

                #sort and trim
                C[i][j][0] = C[i][j][0][:2]
                C[i][j][1] = C[i][j][1][:2]

    return C, C[0][-1][0]

def scorePartialRule(partialRule):
    # gives a score to a list of tree of the form [t_1, ... t_n, "•"], each of which has a score
    #tentativo 1: punteggio medio degli alberi nella lista....
    return sum(t.score for t in partialRule[:-1])/(len(partialRule) - 1)


def ruleFilter(rule, distributed_vector, dtk_generator):
    if rule in rule_cache:
        punteggio_regola = rule_cache[rule]
        return punteggio_regola

    else:
        regola_albero = rule.toTree()
        numero_nodi = len(list(regola_albero.allNodes()))
        dtf_regola = dtk_generator.dtf(regola_albero)
        punteggio_regola = numpy.dot(dtf_regola,distributed_vector)
        #norma_regola = numpy.dot(dtf_regola,dtf_regola)
        norma = numpy.dot(dtk_generator.dtf(regola_albero), dtk_generator.dtf(regola_albero))
        punteggio_regola = punteggio_regola/norma
        rule_cache[rule] = punteggio_regola
        return punteggio_regola


rule_cache = {} #TODO: add caching logic
def cyk_plus_dtk(sentence, grammar, k_best=2, distributed_vector=None, dtk_generator=None, referenceTable=None, rule_filter=2):
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
                rules = grammar.terminalrules[words[i]]
                #if not rules:
                if True:
                    #print ("word, ", i, " non trovata.")
                    for sym in grammar.symbols:
                        rule = gramm.Rule(sym,[words[i]])
                        rt = rule.toTree()

                        score = numpy.dot(dtk_generator.sn(rt), distributed_vector)
                        #score = numpy.dot(dtk_generator.dtf(rt), distributed_vector)
                        #score = sorting_method(dtk_generator, rt, distributed_vector)
                        ## NORMALIZATION
                        score = score/numpy.sqrt(numpy.dot(dtk_generator.sn(rt), dtk_generator.sn(rt)))
                        rt.score = score

                        C[i][j][0].append(rt)

                        #return None, []
                else:
                    for r in rules:
                        rt = r.toTree()
                        #compute score
                        score = numpy.dot(dtk_generator.sn(rt), distributed_vector)
                        numero_dtk = numero_dtk + 1
                        rt.score = score
                        if rt not in C[i][j][0]:
                            C[i][j][0].append(rt)

                C[i][j][0] = sorted(C[i][j][0], key=lambda x: x.score, reverse=True)[:k_best]

                #self-filling part
                #print ("prima: ", len(C[i, j][0]))
                for B in C[i, j][0]:  #B = A -> B C
                    B_string = B.root
                    rules = grammar.nonterminalrules[B_string] #X -> A •

                    for r in rules:
                        if B_string != " ".join(r.right):
                            if [B, "•"] not in C[i, j][1]:    # <- devo dare uno score a questo (o forse no?)
                                C[i, j][1].append([B, "•"])
                        else:
                            new_tree = tree(root=r.left, children=[B])
                            score = numpy.dot(dtk_generator.sn(new_tree), distributed_vector)
                            numero_dtk = numero_dtk + 1
                            #print (score, B.score, score > B.score)
                            if score > B.score: #pensare ad un filtro più stringente....
                                new_tree.score = score
                                #print (new_tree)
                                C[i][j][0].append(new_tree)
                    if len(C[i, j][0]) > 10:
                        break
                #print ("dopo: ", len(C[i][j][0]))

                #sort and trimming (credo che non serva sortare l'altra lista...)
                C[i][j][0] = sorted(C[i][j][0], key=lambda x: x.score, reverse=True)[:k_best]
                C[i][j][1] = sorted(C[i][j][1], key=lambda x: scorePartialRule(x), reverse=True)[:k_best]


            if j > i:
                for k in range(0, j):
                    first_cell_C = C[i, k]
                    second_cell_C = C[k + 1, j]
                    #print (len(first_cell_C[1]), len(second_cell_C[0]))
                    for (x,y) in itertools.product(first_cell_C[1], second_cell_C[0]):
                        xx = " ".join(c.root for c in x[:-1])
                        yy = y.root

                        string = xx + " " + yy

                        rules = grammar.nonterminalrules[string]
                        #print ("regole: ", len(rules), end=" ---- ")

                        for r in rules:
                            #rule filtering
                            regola_albero = r.toTree()
                            numero_nodi = len(list(regola_albero.allNodes()))

                            if ruleFilter(r, distributed_vector, dtk_generator) > numpy.power(dtk_generator.LAMBDA, numero_nodi/2)/rule_filter:
                                if " ".join(r.right) == string:
                                    #print (r, "empty")
                                    children = x[:-1]
                                    children.append(y)
                                    new_tree = tree(root=r.left, children=children)
                                    score = numpy.dot(dtk_generator.sn(new_tree), distributed_vector)
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
                C[i][j][0] = sorted(C[i][j][0], key=lambda x: x.score, reverse=True)[:k_best]

                #self-filling part
                #print ("prima: ", len(C[i, j][0]))
                for B in C[i, j][0]:
                    B_string = B.root #B = A -> B C
                    rules = grammar.nonterminalrules[B_string]
                    for r in rules:
                        #TODO: add another rule filter here?
                        numero_nodi = len(list(regola_albero.allNodes()))
                        if ruleFilter(r, distributed_vector, dtk_generator) > numpy.power(dtk_generator.LAMBDA, numero_nodi/2)/rule_filter:
                            if B_string != " ".join(r.right):
                                if [B, "•"] not in C[i, j][1]:
                                    C[i, j][1].append([B, "•"])
                            else:
                                # per evitare loop infiniti aggiungo un albero solo se il suo score è maggiore di quello precedente
                                new_tree = tree(root=r.left, children=[B])
                                score = numpy.dot(dtk_generator.sn(new_tree), distributed_vector)
                                numero_dtk = numero_dtk + 1
                                if score > B.score: #pensare ad un filtro più stringente (e che sicuro non crei loop infiniti) ??
                                    new_tree.score = score
                                    #if r.left not in [x.root for x in C[i, j][0]]:  # <- pensare bene a sta cosa!!!
                                    C[i][j][0].append(new_tree)
                                    #print ("dopo: ", len(C[i, j][0]), r)
                    if len(C[i, j][0]) > 10: # se ne sto aggiungendo troppi lascio perdere...
                        break
                # sort and trimming (credo che non serva sortare l'altra lista...)
                # pare vada bene a non sortare per niente in questo punto
                C[i][j][0] = sorted(C[i][j][0], key=lambda x: x.score, reverse=True)[:k_best]
                C[i][j][1] = sorted(C[i][j][1], key=lambda x: scorePartialRule(x), reverse=True)[:k_best]

    print (numero_dtk)
    #rendo l'ouput come quello di CYK_easy
    finalList = C[0][-1][0]
    if finalList:
        #final sort (by DTK)
        finalList = sorted(finalList, key=lambda x: numpy.dot(dtk_generator.dt(x),distributed_vector), reverse=True)
        return True, finalList , C
    else:
        #treeToCYKMatrix.printCYKMatrix(simpleTable(P))
        return False, None, C
    #return C, C[0][-1][0]



if __name__ == "__main__":


    #ss = "(S (NP-SBJ (NP (NNP Pierre)(NNP Vinken) )(, ,)(ADJP (NP (CD 61)(NNS years))(JJ old))(, ,))(VP (MD will)(VP (VB join)(NP (DT the)(NN board))(PP-CLR (IN as)(NP (DT a)(JJ nonexecutive)(NN director)))(NP-TMP (NNP Nov.)(CD 29))))(. .))"
    #ss = "(S (INTJ (RB No) )(, ,) (NP-SBJ (PRP it) )(VP (VBD was) (RB n't) (NP-PRD (NNP Black) (NNP Monday) ))(. .) )"
    # s = "(S (NP (NP (NNP john) (NNP snyder)) (, ,) (NP (NP (JJ former) (NN president)) (PP (IN of) (NP (NP (DT the) (NNP los) (NNP angeles) (NN chapter)) (PP (IN of) (NP (NP (NP (DT the) (NNP national) (NNP association)) (PP (IN of) (NP (NNPS investors) (NNP corp.)))) (, ,) (NP (NP (DT an) (NN organization)) (PP (IN of) (NP (NP (NN investment) (NNS clubs)) (CC and) (NP (JJ individual) (NNS investors)))))))))) (, ,)) (VP (VBZ says) (SBAR (S (NP (PRP$ his) (NN fellow) (NN club) (NNS members)) (VP (VP (VBD did) (RB n't) (VP (NN sell) (PP (IN in) (NP (NP (DT the) (NN crash)) (PP (IN of) (NP (CD 1987))))))) (, ,) (CC and) (VP (VB see) (NP (NP (DT no) (NN reason)) (SBAR (S (VP (TO to) (VP (VB sell) (ADVP (RB now)))))))))))) (. .))"
    # t = tree(string=s)
    # print (len(list(t.allNodes())))

    #dtk initialization
    distributed = dtk.DT(dimension=8192, LAMBDA=0.6, operation=operation.fast_shuffled_convolution)

    #one-of grammar creation
    # rules = []
    # for i in range(24):
    #     if i < 10:
    #         string_i = "0" + str(i)
    #     else:
    #         string_i = str(i)
    #     folder = "/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB3/" + string_i
    #     for filename in os.listdir(folder):
    #         if filename.startswith("."):
    #             continue
    #         f = open(os.path.join(folder, filename))
    #         for ss in f.readlines():
    #             l = tree(string=ss)
    #             l = tree.normalize(l)
    #             rules.extend([gramm.fromTreetoRule(x) for x in l.allRules()])
    #
    # g = gramm.Grammar_2(rules)
    # pickle.dump(g, codecs.open("grammar_PLUS.txt", "wb"))
    # print (len(g.nonterminalrules))
    # print (len(g.terminalrules))
    # print (len(g.symbols))

    #grammar loading
    g = pickle.load(open("grammar_PLUS.txt", "rb"))
    # print (len(g.nonterminalrules))
    # print (len(g.terminalrules))
    # print (len(g.symbols))
    #
    # for (i, l) in g.nonterminalrules.items():
    #     if len(i.split(" ")) == 1:
    #         for r in l:
    #             if len(r.right) == 1:
    #                 print (r)




    #dovrei mettere tutto in un readTree generico che accetta qualunque parser

    f = open("/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB3/24/wsj_2415.mrg")
    #f = open("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/pyCYK/alberi_singoli_test.txt")

    #sentence listing
    sents = []
    for ss in f.readlines():
        l = tree(string=ss)
        l = tree.normalize(l)
        if len(list(l.allNodes())) > 100:
            continue
        sents.append((l, tree.sentence_(l)))



    L = sents[:]
    giusti = 0
    for (i, (l, sent)) in enumerate(L[:]):
        rule_cache = {}
        distributed.dt_cache = {}
        distributed.sn_cache = {}
        distributed.dtf_cache = {}
        gc.collect()
        print (i, " --- ", sent)
        print (l)

        #distributed vector computation
        v = distributed.dt(l)

        #call the parser
        b, _, M = cyk_plus_dtk(sent, g, k_best=2, dtk_generator=distributed, distributed_vector=v, rule_filter=2)
        print ("--")
        # print (P)
        if _ is not None:
            for t in _:
                print (t == l, metrics.labeled_precision(t, l), metrics.labeled_recall(t, l), metrics.labeled_fscore(t, l))
                if t == l:
                    giusti = giusti + 1
                    print (giusti, len(L))
                    print ("===")
                    break
                print ("===")


















