'''
Created on 08/giu/2015

@author: Fabio
'''

# import sys
# if sys.platform == 'darwin': # sto sul mio pc
#     sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
# else: #sto deployando su venere
#     sys.path.append("/home/ferrone/pyDTK_sync/src")
import numpy

from pydtk import tree

import grammar

def topRule(t):
    return grammar.Rule(t.root, [x.root for x in t.children])

def comparator(tree, reconstructedMatrix):
    originalMatrix = treeToCYKMatrix(tree)
    for i, _ in numpy.ndenumerate(originalMatrix):
        original, reconstructed_ = originalMatrix[i], reconstructedMatrix[i]

        reconstructed = [(topRule(t), t.score) for t in reconstructed_[0]]
        rec = [r[0] for r in reconstructed]
        for j in original:
            if j not in rec:
                print ("differenza a: ", i, j, rec)

def treeToCYKMatrix(t):
    k = conta_terminali(t)
    cyk_matrix = numpy.zeros((k,k), dtype=object)
    for i, _ in numpy.ndenumerate(cyk_matrix):
        cyk_matrix[i] = []
    markNodesInMatrix(t,cyk_matrix,0)
    return cyk_matrix

def partialTreeMatrix(t):
    k = conta_terminali(t)
    cyk_matrix = numpy.zeros((k,k), dtype=object)
    for i, _ in numpy.ndenumerate(cyk_matrix):
        cyk_matrix[i] = []
    markNodesInMatrix(t,cyk_matrix,0, True)
    return cyk_matrix


def partialRulesMatrix(l):
    k = sum(conta_terminali(t) for t in l)
    partialMatrix = numpy.zeros((k,k), dtype = object)
    for i, _ in numpy.ndenumerate(partialMatrix):
        partialMatrix[i] = []
    markPartialNodes(l, partialMatrix, 0)
    return partialMatrix

def markPartialNodes(list, partialMatrix, displacement):

    if len(list) > 1:
        firsts, last = list[:-1], list[-1]
        contaTerminali = sum(conta_terminali(t_) for t_ in firsts)
        partialMatrix[displacement, sum(conta_terminali(t_) for t_ in list) - 1].append([x.root for x in list])
        markPartialNodes(firsts, partialMatrix, displacement)
        markPartialNodes([last], partialMatrix, displacement + contaTerminali)

    else:
        t = list[0]
        if t.isPreTerminal():
            partialMatrix[displacement, 0].append(t)
        else:
        # if t.children:
            list = t.children
            markPartialNodes(t.children, partialMatrix, displacement)
        # else:
        #     partialMatrix[displacement, 0].append(t)

    return partialMatrix

def conta_terminali(t):
    return len([a for a in t.allTerminalNodes()])

def markNodesInMatrix(t,cky_matrix,displacement, returnTree=False):
    if (t.isPreTerminal()) :
        if returnTree:
            cky_matrix[displacement, 0].append(t)
        else:
            cky_matrix[displacement, 0].append(grammar.Rule(t.root, [x.root for x in t.children]))
    else:
        if returnTree:
            cky_matrix[displacement, conta_terminali(t)-1].append(t)
        else:
            cky_matrix[displacement, conta_terminali(t)-1].append(grammar.Rule(t.root,[x.root for x in t.children]))
        for i, x in enumerate(t.children):
            if i == 0:
                markNodesInMatrix(x, cky_matrix, displacement, returnTree)
            else:
                markNodesInMatrix(x, cky_matrix, displacement + sum(conta_terminali(x) for x in t.children[:i]), returnTree)
        # markNodesInMatrix(t.children[0],cky_matrix,displacement)
        # markNodesInMatrix(t.children[1],cky_matrix,displacement + conta_terminali(t.children[0]))
    return cky_matrix

def printCYKMatrix(cyk_matrix):

    for row in cyk_matrix :
        for element in row:
            print(element , end="\t")
        print()

def spanningTree(t, j, i=0):
    M = partialTreeMatrix(t)
    c = M[i, j]
    if c:
        return c
    else:
        for n in range(1, j):
            a, b = (i, j-n), (j-n+1, n-1)
            print (n, a, b)
            a_ = M[a]
            b_ = M[b]
            if a_ and b_:
                return a_, b_
            else:
                return spanningTree(t, a[1], a[0]) + spanningTree(t, b[1], b[0])

if __name__ == "__main__":
    treeString1 = "(S (CC but) (SBAR (IN while) (S (NP (DT the) (NNP new) (NNP york) (NNP stock) (NNP exchange)) (VP (VBD did) (RB n't) (VP (VB fall) (ADVP (RB apart)) (NP (NNP friday)) (SBAR (IN as) (S (NP (DT the) (NNP dow) (NNP jones) (NNP industrial) (NNP average)) (VP (VBD plunged) (NP (NP (CD 190.58) (NNS points)) (PRN (: --) (NP (NP (JJS most)) (PP (IN of) (NP (PRP it))) (PP (IN in) (NP (DT the) (JJ final) (NN hour)))) (: --)))))))))) (NP (PRP it)) (ADVP (RB barely)) (VP (VBD managed) (S (VP (TO to) (VP (VB stay) (NP (NP (DT this) (NN side)) (PP (IN of) (NP (NN chaos)))))))) (. .))"
    treeString2 = "(S (NP (NP (DT some) (NN circuit) (NNS breakers)) (VP (VBN installed) (PP (IN after) (NP (DT the) (NNP october) (CD 1987) (NN crash))))) (VP (VBD failed) (NP (PRP$ their) (JJ first) (NN test)) (PRN (, ,) (S (NP (NNS traders)) (VP (VBP say))) (, ,)) (S (ADJP (JJ unable) (S (VP (TO to) (VP (VB cool) (NP (NP (DT the) (NN selling) (NN panic)) (PP (IN in) (NP (DT both) (NNS stocks) (CC and) (NNS futures)))))))))) (. .))"
    treeString4 = "(S (INTJ (RB no)) (, ,) (NP (PRP it)) (VP (VBD was) (RB n't) (NP (NNP black) (NNP monday))) (. .))"

    treeString4 = "(S (@S (@S (@S (INTJ no) (, ,)) (NP it)) (VP (@VP (VBD was) (RB n't)) (NP (NNP black) (NNP monday)))) (. .))"
    treeString7 = "(S (NP (NP (CD seven) (NNP big) (NNP board) (NNS stocks)) (: --) (NP (NP (NNP ual)) (, ,) (NP (NNP amr)) (, ,) (NP (NNP bankamerica)) (, ,) (NP (NNP walt) (NNP disney)) (, ,) (NP (NNP capital) (NNP cities/abc)) (, ,) (NP (NNP philip) (NNP morris)) (CC and) (NP (NNP pacific) (NNP telesis) (NNP group))) (: --)) (VP (VP (VBD stopped) (S (VP (VBG trading)))) (CC and) (VP (ADVP (RB never)) (VBD resumed))) (. .))"
    t = tree.Tree(string = treeString4)
    cyk_matrix = treeToCYKMatrix(t)

    printCYKMatrix(cyk_matrix)


    for i in range(7):
        print ('s', spanningTree(t, i))



    # print ("pr:")
    #
    # pr = partialRulesMatrix(t.children)
    # printCYKMatrix(pr)
