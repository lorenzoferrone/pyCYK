# import pickle
# import numpy
#
# from loadPennTree import loadPennTree
# from metrics import labeled_fscore, labeled_precision, labeled_recall
# from cyk import CYK
#
# import sys
# sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
# from tree import Tree as tree
# import dtk



def numberify(t):
    # returns a tree with sequential numbers in place of words
    for i, node in enumerate(t.allTerminalNodes()):
        node.root = str(i)
    return t
