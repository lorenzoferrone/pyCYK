'''
Created on 08/giu/2015

@author: Fabio
'''

import sys
# sys.path.append("/Users/lorenzo/Documents/Programming/Java/pyDTK2/src")
import numpy
from pydtk import tree

import grammar


def treeToCYKMatrix(t):
    k = conta_terminali(t)
    cyk_matrix = numpy.zeros((k,k), dtype=object)
    markNodesInMatrix(t,cyk_matrix,0)
    return cyk_matrix

def conta_terminali(t):
    return len([a for a in t.allTerminalNodes()])

def markNodesInMatrix(t,cky_matrix,displacement):
    if (t.isPreTerminal()) :
        cky_matrix[0,displacement] = t.root
    else:
        cky_matrix[conta_terminali(t)-1,displacement] = grammar.Rule(t.root,[t.children[0].root,t.children[1].root])
        markNodesInMatrix(t.children[0],cky_matrix,displacement)
        markNodesInMatrix(t.children[1],cky_matrix,displacement + conta_terminali(t.children[0]))
    return cky_matrix

def printCYKMatrix(cyk_matrix):

    # for w in t.allTerminalNodes() :
    #     print(w, end="\t")
    # print()

    for row in cyk_matrix :
        for element in row:
            print(element , end="\t")
        print()


if __name__ == "__main__":
    ss = "(S (@S (NP (@NP (@NP (NP (NNP Pierre)(NNP Vinken))(, ,))(ADJP (NP (CD 61)(NNS years))(JJ old)))(, ,))(VP (MD will)(VP (@VP (@VP (VB join)(NP (DT the)(NN board)))(PP (IN as)(NP (@NP (DT a)(JJ nonexecutive))(NN director))))(NP (NNP Nov.)(CD 29)))))(. .))"

    t = tree.Tree(string = ss)
    cyk_matrix = treeToCYKMatrix(t)

    printCYKMatrix(cyk_matrix)
