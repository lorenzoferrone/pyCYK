__author__ = 'lorenzo'

import numpy
import pickle
import grammar
import sys

from loadPennTree import loadPennTree

def listOfrules(Grammar):
    nt_rules  = []

    for r in Grammar.nonterminalrules.items():
        nt_rules.extend(r[1])
    nt_rules = list(set(nt_rules))
    return sorted(nt_rules)


def grammarMatrix(rulesList):
    # rulesList is a list of rule, including terminal rules (postags)
    # first all pos's, then all rules. poss are encoded as Rule('X', ['NONE'])
    n = len(rulesList) #number of non-terminal + terminal rules
    p = len([x for x in rulesList if type(x) == str]) #number of terminal rules (pos)
    M = numpy.zeros((n - p, n - p))
    rules = rulesList[p:]
    for i, rule in enumerate(rules): #cycling only non-terminal rules
        b, c = rule.right
        index = [rules.index(r) for r in rules if r.left == b or r.left == c]
        M[i, index] = 1
        M[i, i] = 1
    return M

def terminalRuleMatrix(rulesList):
    posList = [x for x in rulesList if type(x) == str]
    n = len(rulesList)
    p = len(posList)
    M = numpy.zeros((n - p, p))
    rules = rulesList[p:]
    for i, rule in enumerate(rules): #cycling only non-terminal rules
        b, c = rule.right
        index = [posList.index(pp) for pp in posList if pp == b or pp == c]
        M[i, index] = 1
    return M

def fromPosListToRule(posList):
    return [grammar.Rule(x, ['None']) for x in posList]



if __name__ == '__main__':


    # PennTreePath = "/home/ferrone/Datasets/PTB2/"
    #
    # #return list of trees
    # treeList = list(loadPennTree(PennTreePath, sections=['00']))

    Grammar = pickle.load(open('binaryGrammarMostFrequent1500.txt', 'rb'))

    # rs = [grammar.Rule("A", ['B', 'C'])]
    #
    # Grammar = grammar.Grammar(rs)
    #
    # print (listOfrules(Grammar))
    # print (len(Grammar.posTags))
    # print (fromPosListToRule(Grammar.posTags))
    #
    # sys.exit(0)


    rulesList = listOfrules(Grammar)

    # G = grammarMatrix(rulesList)
    T = terminalRuleMatrix(rulesList)


    # print (G)
    # print (G.shape)

    print (T)
    print (T.shape)

    # g = G[:,:1500]
    # print (g)
    # print (g.shape)
    #
    # print (numpy.linalg.norm(g))

    # numpy.save('grammarMatrix_1', g)
