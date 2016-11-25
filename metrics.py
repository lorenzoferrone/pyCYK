__author__ = 'lorenzo'

import sys

sys.path.append("/Users/lorenzo/Documents/Programming/Java/pyDTK2/src")
from tree import Tree as tree

def precision(t1, t2):
    #first tree is candidate, second is gold standard
    tot = 0
    correct = 0
    gold_span = [x.span() for x in t2.allNodes()]
    for n in t1.allNodes():
        if n.isTerminal():
            continue
        tot = tot + 1

        if n.span() in gold_span:
            correct = correct + 1

    return correct/tot

def labeled_precision(t1, t2):
    #first tree is candidate, second is gold standard
    tot = 0
    correct = 0
    gold_span = [(x.root, x.span()) for x in t2.allNodes()]
    for n in t1.allNodes():
        if n.isTerminal():
            continue
        tot = tot + 1

        if (n.root, n.span()) in gold_span:
            correct = correct + 1

    return correct/tot

def recall(t1, t2):
    #first tree is candidate, second is gold standard
    tot = 0
    correct = 0
    test_span = [x.span() for x in t1.allNodes()]
    for n in t2.allNodes():
        if n.isTerminal():
            continue
        tot = tot + 1

        if n.span() in test_span:
            correct = correct + 1

    return correct/tot

def labeled_recall(t1, t2):
    #first tree is candidate, second is gold standard
    tot = 0
    correct = 0
    test_span = [(x.root, x.span()) for x in t1.allNodes()]
    for n in t2.allNodes():
        if n.isTerminal():
            continue
        tot = tot + 1

        if (n.root, n.span()) in test_span:
            correct = correct + 1

    return correct/tot


def fscore(t1, t2):
    p = precision(t1, t2)
    r = recall (t1, t2)
    return 2*(p*r/(p + r))

def labeled_fscore(t1, t2):
    p = labeled_precision(t1, t2)
    r = labeled_recall (t1, t2)
    return 2*(p*r/(p + r))

def crossbracketing():
    pass



if __name__ == '__main__':
    s1 = "(S (NP (NNS futures) (NNS traders)) (VP (VP say) (SBAR (SBAR (NP (DT the) (NNP s&p)) (VP (VBD was) (VP (VBG signaling) (SBAR (IN that) (S (NP (DT the) (NNP dow)) (VP (MD could) (VP (VB fall) (NP (RB as) (JJ much))))))))) (SBAR (RB as) (FRAG (NP (CD 200) (NNS points)) (. .))))))"
    s2 = "(S (@S (NP (NNS futures) (NNS traders)) (VP (VBP say) (SBAR (NP (DT the) (NNP s&p)) (VP (VBD was) (VP (VBG signaling) (SBAR (IN that) (S (NP (DT the) (NNP dow)) (VP (MD could) (VP (VB fall) (NP (NP (RB as) (JJ much)) (PP (IN as) (NP (CD 200) (NNS points))))))))))))) (. .))"

    s1 = "(W (X a) (Y b) (Z (c) (d)))"
    s2 = "(W (X a) (Y (Z b) (V (c) (d))))"

    t1 = tree(string=s1)
    t2 = tree(string=s2)

    p = precision(t1, t2)
    r = recall(t1, t2)
    f = fscore(t1, t2)


    print ("precision: ", p, "recall: ", r, "fscore: ", f)


    print ("lp: ", labeled_precision(t1, t2))
    print ("lr: ", labeled_recall(t1, t2))
    print ("lf: ", labeled_fscore(t1, t2))