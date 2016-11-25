__author__ = 'lorenzo'

import numpy
import sys
import random
from collections import defaultdict
import pickle
import time
from functools import total_ordering
from collections import Counter
import os

# if sys.platform == 'darwin': # sto sul mio pc
#     sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
# else: #sto deployando su venere
#     sys.path.append("/home/ferrone/pyDTK_sync2/src")

from pydtk.tree import Tree as tree

from loadPennTree import loadPennTree





# def fromTreetoRule(tree):
#     #forse andrebbe spostata dentro Rule
#     children_string = [x.root for x in tree.children]
#     r = Rule(tree.root, children_string)
#     r.terminalRule = tree.terminalRule
#     return r

@total_ordering
class Rule:

    def __init__(self, left, right, prob=1):
        self.left = left
        self.right = right
        self.prob = prob

    @classmethod
    def fromTree(cls, tree):
        children_string = [x.root for x in tree.children]
        r = Rule(tree.root, children_string)
        r.terminalRule = tree.terminalRule
        return r


    def __repr__(self):
        if len(self.right) >= 2:
            return self.left + " -> " + " ".join(self.right)
        else:
            return self.left + " -> " + self.right[0]

    def __str__(self):
        if len(self.right) >= 2:
            return self.left + " -> " + " ".join(self.right)
        else:
            return self.left + " -> " + self.right[0]

    def encode(self):
        return self.left.encode("utf-8")

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __lt__(self, other):
        return self.__str__() < other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def toTree(self):
        children = [tree(x, None) for x in self.right]
        return tree(self.left, children)


class Grammar:

    @classmethod
    def fromTreeBank(cls, pathToTreeBank, sections):
        trees = loadPennTree(pathToTreeBank, sections)
        return cls.fromTrees(trees)

    @classmethod
    def fromTrees(cls, trees, maxRules=None):
        rules = []
        for tree in trees:
            rules.extend([Rule.fromTree(x) for x in tree.allRules()])
        r_ = [r for r in rules if not r.terminalRule]
        t_ = [r for r in rules if r.terminalRule]

        c = Counter(r_)
        if maxRules:
            rules_ = [x[0] for x in c.most_common(maxRules)] + t_
        else:
            rules_ = rules

        return cls(rules_)

    def __init__(self, rules=None):
        if rules is None:
            pass

        rules = list(set(rules))

        #hardcoding is evil but well.
        self.posTags = "N V P DET CC CD DT EX FW IN JJ JJR JJS LS MD NN NNS NNP NNPS PDT POS PRP PRP$ RB RBR RBS RP SYM TO UH VB VBD VBG VBN VBP VBZ WDT WP WP$ WRB . , : ( ) '' `` -- $".split()

        #loop over rules and divide between terminal and not terminal rules, also adding each symbol to their list
        words = set()
        symbols = set()
        terminalrules = defaultdict(list)
        nonterminalrules = defaultdict(list)
        for i, rule in enumerate(rules):
            symbols.add(rule.left) #add left symbol in any case
            if len(rule.right) == 1:
                word = rule.right[0] #not necessarily a word, a symbol which appears alone as a right-hand side of a rule
                if rule.terminalRule:
                    words.add(word)
                    terminalrules[word].append(rule)
                else:
                    symbols.add(word)
                    nonterminalrules[word].append(rule)

            #TODO: modificare per regole con piu di 2 elementi (sembra ok)
            if len(rule.right) >= 2:
                for s in rule.right:
                    symbols.add(s)
                righthandside = " ".join(rule.right)
                if rule not in nonterminalrules[righthandside]:
                    nonterminalrules[righthandside].append(rule)
                #nonterminalrules[rule.right[1]].append(rule)

                if len(rule.right) >= 2:
                    # questa parte solo nelle regole nuove, così quelle binarie non cambia niente
                    # aggiungo alla grammatica tutte le regole che iniziano con una certa sequenza
                    for split in range(1, len(rule.right) + 1):

                        rule_string = " ".join(rule.right[:split])
                        #print (righthandside, "-", rule_string)
                        if rule not in nonterminalrules[rule_string]:
                            nonterminalrules[rule_string].append(rule)


        #transform symbol in a dict of symbol:index
        symbols_ = list(symbols)
        self.symbols = {symbol: index for symbol, index in zip(symbols_, range(len(symbols_)))}
        self.indexes = {index: symbol for (symbol, index) in self.symbols.items()}

        self.nonterminalrules = nonterminalrules
        self.terminalrules = terminalrules

        self.rules = self.nonterminalrules.copy()
        self.rules = self.rules.update(self.terminalrules)


    def checkCoverage(self, t):
        rules = [Rule.fromTree(r) for r in t.allRules()]
        tot = True
        regole_non_trovate = []
        for x in rules:
            if len(x.right) == 1 and x.terminalRule:
                parola = x.right[0]
                if x in self.terminalrules[parola]:
                    continue
                else:
                    regole_non_trovate.append(x)
                    tot = False
            else:
                string = x.right
                string = " ".join(string)
                if x in self.nonterminalrules[string]:
                    continue
                else:
                    regole_non_trovate.append(x)
                    tot = False

        return tot, regole_non_trovate


if __name__ == "__main__":


    # Grammar = pickle.load(open("binaryGrammar.txt", "rb"))

    Grammar = pickle.load(open("fullGrammarNormalized.txt", "rb"))      #full grammar

    t_rules = []
    nt_rules  = []
    for r in Grammar.nonterminalrules.items():
        # print (r)
        t_rules.extend(r[1])

    for r in Grammar.terminalrules.items():
        nt_rules.extend(r[1])

    t_rules = list(set(t_rules))
    nt_rules = list(set(nt_rules))

    t_rules.sort(key=lambda rule: rule.left)
    nt_rules.sort(key=lambda rule: rule.left)

    file = open("/Users/lorenzo/Desktop/NLPFullGrammar.txt", "w")

    for rr in t_rules + nt_rules:
        file.write(rr.__str__())
        file.write('\n')


    print (len(t_rules))
    print (len(nt_rules))
