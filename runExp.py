__author__ = 'lorenzo'


import numpy
import sys, getopt


# if sys.platform == 'darwin': # sto sul mio pc
#     sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
# else: #sto deployando su venere
#     sys.path.append("/home/ferrone/pyDTK_sync/src")

import random
import pickle
import time
# import gc
# import seaborn as sns
# import matplotlib.pyplot as plt
import itertools

from pydtk.tree import Tree as tree
# import pydtk.dtk as dtk
# from pydtk import dtk
# import pydtk.dtk2 as dtk2
from pydtk import dtk2

import grammar
from metrics import fscore, precision, recall, labeled_fscore, labeled_precision, labeled_recall
import treeToCYKMatrix
from cyk import CYK
# from cykPlus import CYKPlus
from cykPlus2 import CYKPlus
from loadPennTree import loadPennTree
# from .grammar import Grammar as grammar

from cykWithNumber import numberify


def distort(v, epsilon=0.001):
    l = len(v)
    w = epsilon*numpy.random.normal(size=l, scale=1/numpy.sqrt(l))
    return v + w


def runExp(treeList, parser, k_best, k_max, matrix=None, DEBUG=False, distortRate=0):

    if matrix is None:
        distributed = dtk2.DT(dimension=parser.dimension, LAMBDA=parser.LAMBDA, operation=parser.operation)


    giusti = 0          #number of correctly reconstructed trees
    giusti_k = 0        #number of correctly reconstructed tree among the first k positions
    parsati = 0         #number of trees that get parsed (correctly or not)
    uncovered = 0       #number of tree that are not covered by the Grammar
    giusti_tra_tutti = 0

    lista_dati = []     #list of results to analyze

    for i, alb in enumerate(treeList, 1):
        # print ("***\n")
        #resetting caches and garbage collecting to be sure there are no memory leaks.
        parser.cleanCache()

        covered, regole_ = parser.grammar.checkCoverage(alb)    #forse serve solo per DEBUG

        if not covered:
            if not all(r.terminalRule for r in regole_):
                uncovered = uncovered + 1
                lista_dati.append((alb, "uncovered"))
                print ("not covered: ", uncovered)
                print ([(r, r.terminalRule) for r in regole_])
                print (any(r.terminalRule for r in regole_))
                continue

        sent = tree.sentence_(alb)

        # if DEBUG:
        #     for rule in alb.allRules():
        #         print (rule, numpy.dot(distributed.dt(alb), distributed.dt(rule)))


        # commento la parte del printing
        print (i, ": parsing: ", sent)
        # print (alb)
        print ('.', end='', flush=True)

        for j in range(k_best, k_max + 1):

            # if DEBUG:
            #     referenceTable = treeToCYKMatrix.treeToCYKMatrix(alb)
            #     isParsed, treeList, CYKMatrix = parser.parser_with_reconstruction3(sent, G, j, distributed.dt(alb), distributed, referenceTable=referenceTable, rule_filter=filter)

            #else:
            if matrix is None:
                # if there is no matrix of vectors, use the distributed class to compute distributed vector
                # try distorting vector with noise
                v = distort(distributed.dt(alb), distortRate)
                # v = v/numpy.linalg.norm(v)
                # v = 3*v
                isParsed, parseList, CYKMatrix = parser.parse(sent, j, v)
            else:
                # otherwise I use the matrix itself (indexes starts from 0)
                isParsed, parseList, CYKMatrix = parser.parse(sent, j, matrix[i-1])

            if isParsed:
                break

        if isParsed:
            parsati = parsati + 1
            #print ("OK", j) #tiene conto di quanti tentativi ha fatto
            bestTree = parseList[0]
            # print (bestTree)
            # print (bestTree == alb)

            if bestTree == alb:
                giusti = giusti + 1

            for otherTree in parseList:
                if otherTree == alb:
                    giusti_k = giusti_k + 1
                    break

            #map(lambda f: f(bestTree, alb), [labeled_precision, labeled_recall, labeled_recall])
            lista_dati.append((alb, "parsed", j,
                               labeled_precision(bestTree, alb),
                               labeled_recall(bestTree, alb),
                               labeled_fscore(bestTree, alb)))

        else:
            #print ("not parsed!")
            lista_dati.append((alb, "not_parsed", j, 0, 0, 0))



        # print running statistics
        print ("Giusti ", giusti, " Giusti k ", giusti_k, "parsati", parsati , "covered", i - uncovered, "totali", i)

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

        # print ("PARSED prec, rec, fscore:\t\t", numpy.mean(ps_parsed), numpy.mean(rs_parsed), numpy.mean(fs_parsed))
        # print ("COVERED running prec, rec, fscore:\t", numpy.mean(ps_covered), numpy.mean(rs_covered), numpy.mean(fs_covered))

    i = len(treeList)
    covered = i - uncovered
    # print ("\n##RIASSUNTO##\n")
    print ("\nparametri: lambda={0}, dimension={1}, k={2}, k_max={3}, filter={4}".format(parser.LAMBDA, parser.dimension, k_best, k_max, filter))
    print ("numero frasi: ", i)
    print ("covered: ", covered, 100*covered/i,"%")
    print ("parsati: ", parsati, 100*parsati/covered,"%")
    print ("corretti: ", giusti, 100*giusti/parsati,"%")
    print ("corretti a k: ", giusti_k, 100*giusti_k/parsati,"%")
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
    i = len(lista_dati)
    parsed = [x for x in lista_dati if x[1] == "parsed"]
    # l_parsed = len(parsed)
    not_parsed = [x for x in lista_dati if x[1] == "not_parsed"]
    # l_not_parsed = len(not_parsed)
    covered = [x for x in lista_dati if x[1] != "uncovered"]
    l_covered = len(covered)
    uncovered = i - len(covered)
    ps_parsed = [x[3] for x in parsed]
    rs_parsed = [x[4] for x in parsed]
    fs_parsed = [x[5] for x in parsed]

    ps_covered = [x[3] for x in covered]
    rs_covered = [x[4] for x in covered]
    fs_covered = [x[5] for x in covered]

    # print ("numero frasi: ", i)
    # print ("covered: ", l_covered, 100*l_covered/i,"%")
    # print ("parsati: ", l_parsed, 100*l_parsed/covered,"%")
    # print ("corretti: ", giusti, 100*giusti/parsati,"%")
    # print ("corretti a k: ", giusti_k, 100*giusti_k/parsati,"%")
    # print ("corretti totali: ", giusti, 100*giusti/i,"%")

    print ("PARSED: ")
    print ("precision: ", numpy.mean(ps_parsed))
    print ("recall: ", numpy.mean(rs_parsed))
    print ("f-score: ", numpy.mean(fs_parsed))
    print ("COVERED:")
    print ("precision: ", numpy.mean(ps_covered))
    print ("recall: ", numpy.mean(rs_covered))
    print ("f-score: ", numpy.mean(fs_covered))

    print ("normalized recall: ", numpy.mean(rs_parsed)*len(parsed)/len(lista_dati))
    print ('\n')


if __name__ == '__main__':

    # arguments:
    # -b for binary mode
    # -f for full
    # -n for number mode
    # -i <file> for vector_file (optional) if not present default to dtk of input tree
    # TODO: -p for pennTreeBank path
    # TODO: -s for pennTreeBank sections
    # TODO: -g for grammar file path

    # launch as python3 runExp.py [options]


    vector_file = None
    NUMBERIFY = False

    options, arguments = getopt.getopt(sys.argv[1:], "bfni:", [])
    for opt, arg in options:
        if opt == "-n":
            NUMBERIFY = True
        if opt == "-b":
            MODE = "binary"
        if opt == '-f':
            MODE = "full"
        if opt == "-i":
            vector_file = arg


    print (MODE, NUMBERIFY, vector_file)

    MATRIX = numpy.load(vector_file) if vector_file else None

    # some old vector_file
    # vector_file = '/home/ferrone/dtknn/EXPE/8192_512_512_matrix/test_pred.512.512.1.npy'
    # vector_file = "/home/ferrone/dtknn/KerasVersion/reconstructed.npy"
    # vector_file = "/home/ferrone/dtknn/BinaryMatrice/8192.0.6/pennTreeMatrix.section.23.npy"
    # vector_file = "/home/ferrone/dtknn/KerasVersion/reconstructed_from_encoded.npy"
    # vector_file = "/home/ferrone/dtknn/KerasVersion/reconstructed_number_lstm.npy"
    # vector_file = '/home/ferrone/dtknn/KerasVersion/reconstruced_lstm_numberify_from_postags.npy'
    # vector_file = '/home/ferrone/dtknn/KerasVersion/semantic_reconstruced_lstm.npy'
    # vector_file = "/home/ferrone/dtknn/KerasVersion/reconstruced_lstm_numberify.npy"


    if MODE == "binary":
        # PennTreePath = "/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB2/"
        PennTreePath = "/home/ferrone/Datasets/PTB2/"
        Grammar = pickle.load(open("./Grammars/binaryGrammar23.txt", "rb"))
    else:
        # PennTreePath = "/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB3/"
        PennTreePath = "/home/ferrone/Datasets/PTB3/"
        Grammar = pickle.load(open("./Grammars/fullGrammarNormalized.txt", "rb"))      #full grammar

    # file format is /path/XX/wsjxxyy.mrg where XX goes from 00 to 24 and yy from 00 to 99
    # (except for section 24 where yy goes to 54)

    # Grammar Creation (comment or uncomment as needed)
    # sections = ["0" + str(x) for x in range(10)] + [str(x) for x in range(10, 24)] #>> ["00", "01", ..., "23"]
    # treesForGrammar = loadPennTree(PennTreePath, sections, normalize=False)
    # Grammar = grammar.Grammar.fromTrees(treesForGrammar)
    # print (Grammar.terminalrules)
    # if MODE == "binary":
    #     # and then save it
    #     print ('saving')
    #     pickle.dump(Grammar, open("./Grammars/binaryGrammar23.txt", "wb"))
    # else:
    #     print ('saving')
    #     pickle.dump(Grammar, open("./Grammars/fullGrammarNormalized.txt", "wb"))


    # print (len(Grammar.nonterminalrules))
    # print (len(Grammar.symbols))

    # s = set()
    # max_length = 0
    # for k, rs in Grammar.nonterminalrules.items():
    #     for r in rs:
    #         s.add(r)
    #         m = len(r.right)
    #         if m == 1:
    #             print (m, r)
    #             sys.exit(0)
    # print (len(s))
    #
    # sys.exit(0)

    sections = ["23"]
    if MODE == "binary":
        if NUMBERIFY:
            # numberified version
            treeList = [numberify(t) for t in list(loadPennTree(PennTreePath, sections, normalize=True))]
        else:
            treeList = list(loadPennTree(PennTreePath, sections, normalize=True))


    else:
        if NUMBERIFY:
            treeList = [numberify(t) for t in list(loadPennTree(PennTreePath, sections, normalize=False))]
        else:
            treeList = list(loadPennTree(PennTreePath, sections, normalize=False))


    print (len(treeList))
    # print (MATRIX.shape)
    print (treeList[0])

    # run the exps cycling through different parameters
    for dimension in [8192]:
        # print ('dimension: ', dimension)
        for LAMBDA in [0.6]:
            for filter in [1.5]:
                # print ('filter: ', filter)

                # creating parser instance
                if MODE == "binary":
                    cykInstance = CYK(dimension, LAMBDA, Grammar, filter=filter)
                else:
                    cykInstance = CYKPlus(dimension, LAMBDA, Grammar, filter=filter)


                # creating or loading the appropriate matrix of distributed trees
                # matrix = pickle.load("matrixFile", "rb")
                for distortRate in [0]:
                    results = runExp(treeList, cykInstance, k_best=2, k_max=2, matrix=MATRIX, distortRate=distortRate)
                    average_metrics(results)
        print ('###')



                #pickle.dump(results, open("lista_risultati.txt", "wb"))
