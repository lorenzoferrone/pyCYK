__author__ = 'lorenzo'


import numpy
import sys

sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
# sys.path.append("/home/ferrone/pyDTK/src")
import random
import pickle
import time
import gc
#import seaborn as sns
import matplotlib.pyplot as plt

import itertools

from tree import Tree as tree
import grammar
import cyk_easy
import dtk
import operation
from metrics import fscore, precision, recall, labeled_fscore, labeled_precision, labeled_recall
import treeToCYKMatrix

# def fromTreetoRule(tree):
#     children_string = tuple([x.root for x in tree.children])
#     return grammar.Rule(tree.root, children_string)

def runExp(lista_alberi, grammar, k_best, k_max, distributed, DEBUG=False, filter=2):
    giusti = 0      #number of correctly reconstructed trees
    giusti_k = 0    #number of correctly reconstructed tree among the first k positions
    parsati = 0     #number of trees that get parsed (correctly or not)
    uncovered = 0   #number of tree that are not covered by the Grammar
    giusti_tra_tutti = 0

    lista_dati = []

    for i, alb in enumerate(lista_alberi, 1):
        print ("***\n")
        #resetting caches and garbage collecting to be sure there are no memory leaks.
        distributed.dt_cache = {}
        distributed.sn_cache = {}
        distributed.dtf_cache = {}
        cyk_easy.rule_cache = {}
        gc.collect()
        # covered, regole_ = grammar.checkCoverage(alb)
        #
        # if not covered:
        #     uncovered = uncovered + 1
        #     lista_dati.append((alb, "uncovered"))
        #     print ("not covered")

        sent = tree.sentence_(alb)  #TODO mettere apposto sta cosa

        if DEBUG:
            for rule in alb.allRules():
                print (rule, numpy.dot(distributed.dt(alb), distributed.dt(rule)))

        print (i, ": parsing: ", sent)
        print (alb)

        for j in range(k_best, k_max + 1):

            if DEBUG:
                referenceTable = treeToCYKMatrix.treeToCYKMatrix(alb)
                isParsed, treeList, CYKMatrix = cyk_easy.parser_with_reconstruction3(sent, G, j, distributed.dt(alb), distributed, referenceTable=referenceTable, rule_filter=filter)

            else:
                isParsed, treeList, CYKMatrix = cyk_easy.parser_with_reconstruction3(sent, G, j, distributed.dt(alb), distributed, rule_filter=filter)

            if isParsed:
                break

        if isParsed:
            parsati = parsati + 1
            #print ("OK", j) #tiene conto di quanti tentativi ha fatto
            bestTree = treeList[0]
            print (bestTree)
            print (bestTree == alb)

            if bestTree == alb:
                giusti = giusti + 1


            for otherTree in treeList:
                if otherTree == alb:
                    giusti_k = giusti_k + 1

            
            #map(lambda f: f(bestTree, alb), [labeled_precision, labeled_recall, labeled_recall])
            lista_dati.append((alb, "parsed", j, labeled_precision(bestTree, alb), labeled_recall(bestTree, alb), labeled_fscore(bestTree, alb)))

        else:
            print ("not parsed!")
            lista_dati.append((alb, "not_parsed", j, 0, 0, 0))
        if DEBUG:
            treeToCYKMatrix.printCYKMatrix(cyk_easy.simpleTable(CYKMatrix))
            treeToCYKMatrix.printCYKMatrix(referenceTable)

        print ("Giusti ", giusti, " Giusti k ", giusti_k, "parsati", parsati , "totali", i)
        #running precision, recall and fscore?
        parsed = [x for x in lista_dati if x[1] == "parsed"]
        not_parsed = [x for x in lista_dati if x[1] == "not_parsed"]
        covered = [x for x in lista_dati if x[1] != "uncovered"]
        ps_parsed = [x[3] for x in parsed]
        rs_parsed = [x[4] for x in parsed]
        fs_parsed = [x[5] for x in parsed]

        ps_covered = [x[3] for x in covered]
        rs_covered = [x[4] for x in covered]
        fs_covered = [x[5] for x in covered]

        print ("PARSED prec, rec, fscore:\t\t", numpy.mean(ps_parsed), numpy.mean(rs_parsed), numpy.mean(fs_parsed))
        print ("COVERED running prec, rec, fscore:\t", numpy.mean(ps_covered), numpy.mean(rs_covered), numpy.mean(fs_covered))

    covered = i - uncovered
    print ("\n##RIASSUNTO##\n")
    print ("parametri: lambda={0}, dimension={1}, k={2}, k_max={3}, filter={4}".format(distributed.LAMBDA, distributed.dimension, k_best, k_max, filter))
    print ("numero frasi: ", i)
    print ("covered: ", covered, 100*covered/i,"%")
    print ("parsati: ", parsati, 100*parsati/covered,"%")
    print ("corretti: ", giusti, 100*giusti/parsati,"%")
    print ("corretti a k: ", giusti, 100*giusti_k/parsati,"%")
    print ("corretti totali: ", giusti, 100*giusti/i,"%")

    return lista_dati

def plotResults(lista_dati):
    print (lista_dati)
    uncovered = [x for x in lista_dati if x[1] == "uncovered"]
    parsed = [x for x in lista_dati if x[1] == "parsed"]
    not_parsed = [x for x in lista_dati if x[1] == "not_parsed"]

    covered = [x for x in lista_dati if x[1] != "uncovered"]

    # for t in parsed:
    #     print (t[0].sentence, len(t[0].sentence.split()))

    covered = sorted(covered, key=lambda t: len(t[0].sentence.split()))

    # ps = [x[3] for x in parsed]
    # rs = [x[4] for x in parsed]
    # fs = [x[5] for x in parsed]

    by_length = []
    length = []
    for k,g in itertools.groupby(covered, key=lambda t: len(t[0].sentence.split())):
        by_length.append(list(g))
        length.append(k)

    print (by_length)
    print (length)

    plot_by_length_ps = [numpy.mean([x[3] for x in y]) for y in by_length]
    plot_by_length_rs = [numpy.mean([x[4] for x in y]) for y in by_length]
    plot_by_length_fs = [numpy.mean([x[5] for x in y]) for y in by_length]



    plt.plot(length, plot_by_length_ps, length, plot_by_length_rs, length, plot_by_length_fs)
    plt.show()

    # for x in [ps, rs, fs]:
    #     print (len(x), min(x), max(x), numpy.mean(x))
    #     print (x)
    #     plt.plot(x)
    #     plt.show()

def average_metrics(lista_dati):
    parsed = [x for x in lista_dati if x[1] == "parsed"]
    not_parsed = [x for x in lista_dati if x[1] == "not_parsed"]
    covered = [x for x in lista_dati if x[1] != "uncovered"]
    ps_parsed = [x[3] for x in parsed]
    rs_parsed = [x[4] for x in parsed]
    fs_parsed = [x[5] for x in parsed]

    ps_covered = [x[3] for x in covered]
    rs_covered = [x[4] for x in covered]
    fs_covered = [x[5] for x in covered]

    print ("PARSED: ")
    print ("precision: ", numpy.mean(ps_parsed))
    print ("recall: ", numpy.mean(rs_parsed))
    print ("f-score: ", numpy.mean(fs_parsed))
    print ("COVERED:")
    print ("precision: ", numpy.mean(ps_covered))
    print ("recall: ", numpy.mean(rs_covered))
    print ("f-score: ", numpy.mean(fs_covered))

    print ("normalized recall: ", numpy.mean(rs_parsed)*len(parsed)/len(lista_dati))


if __name__ == '__main__':

    #load files

    #gs = [open("not_correctly_reconstructed_2100.txt")]
    #gs = [open("wsj_2100.mrgbinarized.txt")]
    #gs = [open("/Users/lorenzo/Desktop/Current/cyk_results/frase_non_parsa.txt")]
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


    print (len(lista_alberi))
    G = pickle.load(open("grammarPennTree5.txt", "rb"))
    # lista_regole = []
    # for albero in lista_alberi:
    #     for regola in albero.allRules():
    #         lista_regole.append(grammar.fromTreetoRule(regola))


    #G = grammar.Grammar_2(lista_regole)
    print (len(G.nonterminalrules))
    print (len(G.symbols))

    for dim in [8192]:
        for lam in [0.6]:
            for fil in [2, 2.5]:
                distributed = dtk.DT(dimension=dim, LAMBDA=lam, operation=operation.fast_shuffled_convolution)


        # debug_file = open("/Users/lorenzo/Desktop/debug_file.txt", "w")
        # for line in lista_alberi[:15]:
        #     debug_file.write(str(line))
        #     debug_file.write("\n")



                l = runExp(lista_alberi, G, 2, 2, distributed, DEBUG=False, filter=fil)
                average_metrics(l)



    #pickle.dump(l, open("lista_risultati.txt", "wb"))


