import sys
if sys.platform == 'darwin': # sto sul mio pc
    sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
else: #sto deployando su venere
    sys.path.append("/home/ferrone/pyDTK_sync/src")
from tree import Tree as tree
import dtk2
import os

import pickle
import numpy

from loadPennTree import loadPennTree
from metrics import labeled_fscore, labeled_precision, labeled_recall
from cyk import CYK

from cykWithNumber import numberify


# # parsing and saving to file
def parsingAndSaving(listToParse, outputFile, parser):
    distributed = dtk2.DT(dimension=parser.dimension, LAMBDA=parser.LAMBDA, operation=parser.operation)
    with open(outputFile, 'w') as f:
        for i, b in enumerate(list(listToParse)):
            sent = tree.sentence_(b)
            print ('.', end='', flush=True)
            _, parseList, __ = parser.parse(sent, 2, distributed.dt(b))
            parser.cleanCache()
            if parseList:
                f.write(parseList[0].__str__() + "\n")
            else:
                f.write("None \n")

# # and comparing
def comparing(unbinarizedFile, listToCompare):
    prec = []
    rec = []
    fscore = []
    uguali = 0
    n = len(listToCompare)
    not_parsed = 0
    with open(unbinarizedFile, 'r') as db:
        for (dbt, ot) in zip(db.readlines(),listToCompare):
            if not dbt.startswith('('):
                #non è un albero
                not_parsed = not_parsed + 1
                # print ('not tree')
                continue
            else:
                dbt_ = tree(string = dbt.strip('\n'))
                if dbt_ == ot:
                    uguali = uguali + 1

                prec.append(labeled_precision(dbt_, ot))
                rec.append(labeled_recall(dbt_, ot))
                fscore.append(labeled_fscore(dbt_, ot))

    rec_ = (n-not_parsed)/(n)*numpy.mean(rec)
    prec_ = numpy.mean(prec)
    f_ = numpy.mean(fscore)

    return uguali/n, uguali/(n-not_parsed), prec_, rec_, f_


if __name__ == '__main__':


    if sys.platform == 'darwin':
        PennTreePathFull = "/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB3/"
        PennTreePathBinarized = "/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB2/"
        javaString = "java -classpath /Users/lorenzo/Documents/Programming/Java/StanfordParser/target/classes main"
        # path = '/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/pyCYK/'
    else:
        PennTreePathFull = "/home/ferrone/Datasets/PTB3/"
        PennTreePathBinarized = "/home/ferrone/Datasets/PTB2/"
        javaString = "java -classpath /home/ferrone/pyCYK_sync/java_debinerizer/classes main"



    # loading binarized tree to reconstruct, and full tree to compare to
    sections = ['23']
    treeListFull = [numberify(t) for t in list(loadPennTree(PennTreePathFull, sections, normalize=True))[:500]]
    treeListBinarized = [numberify(t) for t in list(loadPennTree(PennTreePathBinarized, sections, normalize=True))[:500]]


    # # loading grammar (binarized)
    # Grammar = pickle.load(open("binaryGrammar.txt", "rb"))
    Grammar = pickle.load(open("binaryGrammar23.txt", "rb"))
    #
    # # defining parser and dtk parameters
    # dimension = 1024
    # filter = 1.5
    # LAMBDA = 0.6

    # path = '/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/pyCYK/'

    for dimension in [1024, 2048, 4096, 8192, 16384]:
        for LAMBDA in [0.6]:
            for filter in [1.5, 2, 2.5]:
                parser = CYK(dimension, LAMBDA, Grammar, filter=filter)

                # filename
                # f_bin = 'binaryoutput_{0}_{1}'.format(dimension, filter)
                # f_debin = 'debinarized_{0}_{1}'.format(dimension, filter)

                # # parsing and saving
                parsingAndSaving(treeListBinarized, "binaryreconstructed.txt", parser)

                ## loading java from python -> procudes debinarized.txt
                os.system(javaString)

                print ('\n parametri: dimensione={0}, filter={1}'.format(dimension, filter))

                # # loading from file (after java unbinarizing) and comparing with original trees
                esatti, esatti_, prec, rec, fscore = comparing('debinarized.txt', treeListFull)

                print ('esatti/totali\t: ', esatti)
                # print ('esatti/parsed\t: ', esatti_)
                print ("precision\t: ", prec)
                print ("recall\t: ", rec)
                # print ("fscore\t: ", fscore)
