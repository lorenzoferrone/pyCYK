__author__ = 'lorenzo'

import numpy
import sys
import itertools

sys.path.append("/Users/lorenzo/Documents/Programming/Java/pyDTK2/src")
#sys.path.append("/home/ferrone/pyDTK2/src")
import random
import dtk
import operation
import pickle

import grammar
from tree import Tree as tree
from treekernel import TreeKernel as TK
import treeToCYKMatrix


# class grammar:
#     #ho capito che la versione che viene usata è l'altra!!!!!
#
#     def __init__(self, rules=None, **kwargs):
#         if rules is None:
#             n_rules = kwargs["n_rules"]
#             n_symbol = kwargs["n_symbol"]
#             rules = self.randomGrammar(n_rules, n_symbol)
#
#         self.rules = rules
#         self.nonterminalsymbol = list(set([x.left for x in rules]))  #elimina i duplicati e ritrasforma in lista
#         #self.start = 0
#         self.nonterminalrules = [x for x in rules if len(x.right) == 2]
#         self.terminalrules = [x for x in rules if len(x.right) == 1]
#         self.terminalsymbol = [x.right[0] for x in self.terminalrules]
#
#         self.start_rule = self.rules[0]
#         self.start_symbol = self.start_rule.left
#
#
#         self.start_symbol_index = self.nonterminalsymbol.index(self.start_symbol)
#         self.rules = self.nonterminalrules + self.terminalrules
#         self.symbol = self.nonterminalsymbol + self.terminalsymbol
#
#         #print (self.nonterminalsymbol, self.start_symbol_index)
#
#     def randomGrammar(self, n_rules=5, n_symbol=5):
#         terminalsymbol = "a b c d e f g h i j k l m n o p q r s t u v z".split()[:]
#         nonterminalsymbol = "a b c d e f g h i j k l m n o p q r s t u v z".upper().split()[:n_symbol]
#         # print (terminalsymbol)
#         # print (nonterminalsymbol)
#         #        symbol = terminalsymbol.extend(nonterminalsymbol)
#         rules = []
#         for i in range(n_rules):
#             left = random.choice(nonterminalsymbol)
#             r1 = random.choice(nonterminalsymbol)
#             r2 = random.choice(nonterminalsymbol)
#             if r1 != r2:
#                 right = (r1, r2)
#             else:
#                 continue
#             rules.append(rule(left, right))
#         production = []
#         opensymbol = []
#         for r in rules:
#             #print ("rule: ", r)
#             opensymbol.extend([r.right[0], r.right[1]])
#         opensymbol = set(opensymbol)
#         #print ("open: ", opensymbol)
#         for s in opensymbol:
#             production.append(rule(s, (random.choice(terminalsymbol)), ))
#             production.append(rule(s, (random.choice(terminalsymbol)), ))
#
#         rules.extend(production)
#         rules = set(rules)
#         return list(rules)
#
#     def parsableString(self, start_rule, depth=3):
#
#         if depth == 0:
#             rule1 = random.choice([x for x in self.terminalrules if x.left == start_rule.right[0]])
#             rule2 = random.choice([x for x in self.terminalrules if x.left == start_rule.right[1]])
#             sr1 = rule(rule1.right[0],("__STOP__",))
#             sr2 = rule(rule2.right[0],("__STOP__",))
#             print ("zero: ", rule1, rule2)
#             t = tree(start_rule, [tree(rule1, [tree(sr1, None),   tree(rule2, [tree(sr2, None)])])])
#
#
#         #start_symbol = start_rule.left
#         if start_rule in self.nonterminalrules:
#             for i in start_rule.right:
#
#                 f = random.choice([0,1])
#                 if f:
#                     possible_rules = [x for x in self.nonterminalrules if x.left == i] #lista di regole non terminali
#
#                     if possible_rules:
#                         rule1 = random.choice(possible_rules)
#                         rule2 = random.choice(possible_rules)
#                         print (rule1, rule2)
#                         t = tree(start_rule, [self.parsableString(rule1, depth=depth-1), self.parsableString(rule2, depth=depth-1)])
#                     #return t
#
#                     else: #provo con una regola terminale
#                         rule1 = random.choice([x for x in self.terminalrules if x.left == i])
#                         sr = rule(rule1.right[0],("__STOP__",))
#                         t = tree(start_rule, [tree(rule1, [tree(sr, None)])])
#
#                 else:
#                     rule1 = random.choice([x for x in self.terminalrules if x.left == i])
#                     sr = rule(rule1.right[0],("__STOP__",))
#                     t = tree(start_rule, [tree(rule1, [tree(sr, None)])])
#
#
#         else:
#             rule1 = random.choice([x for x in self.terminalrules if x.left == i])
#             sr = rule(rule1.right[0],("__STOP__",))
#             t = tree(start_rule, [tree(rule1, [tree(sr, None)])])
#
#         return t


# class rule:
#     def __init__(self, left, right, prob=1):
#
#         self.left = left
#         self.right = right
#         self.prob = prob
#
#     def __repr__(self):
#         if len(self.right) == 2:
#             return self.left + " -> " + " ".join(self.right)
#         else:
#             return self.left + " -> " + self.right[0]
#
#     def __str__(self):
#         if len(self.right) == 2:
#             return self.left + " -> " + " ".join(self.right)
#         else:
#             return self.left + " -> " + self.right[0]
#
#     def encode(self):
#         return self.left.encode("utf-8")
#
#     def __eq__(self, other):
#         return self.__str__() == other.__str__()
#
#     def __hash__(self):
#         return hash(self.__str__())

# class tree:
#     """problema: ora come ora è implementato come un albero di 'regole', ogni nodo è una regola, non una stringa
#
#     DEVO CAMBIARLO ASSOLUTAMENTE
#
#     """
#
#     def __init__(self, root, children):
#         self.root = root
#         self.children = children
#
#         if children is None:
#             self.rule = grammar.Rule(self.root, None) #NEW
#             #self.root.prob = 1
#             self.rule.prob = 1  #NEW
#             self.score = 1
#
#         else:
#             self.rule = grammar.Rule(self.root, tuple(self.children)) #NEW
#             #self.score = self.root.prob * numpy.prod([c.root.prob for c in self.children])
#             #self.score = numpy.prod([c.root.prob for c in self.allNodes()])
#             #self.score = numpy.prod([c.rule.prob for c in self.allNodes()]) #NEW
#             self.score = 1
#
#     def sentence(self):
#         return " ".join(c.root.left for c in self.allNodes() if c.isTerminal())
#
#     def __helper_str__(self):
#         if self.isTerminal():
#             #print (self.root.left)
#             return self.root.left
#         if self.isPreTerminal():
#             return self.root.left + " (" + " ".join(c.__helper_str__() for c in self.children) + ")"
#
#         else:
#             return self.root.left + " (" + " ".join(c.__helper_str__() for c in self.children) + ")"
#
#     def __str__(self):
#         return "(" + self.__helper_str__() + ")"
#
#
#     def __eq__(self, other):
#         return self.__str__() == other.__str__()
#
#     def __hash__(self):
#         return hash(self.__str__())
#
#
#     def add(self, children):
#         self.children = children
#
#     def isTerminal(self):
#         return self.children is None
#
#     def isPreTerminal(self):
#         if self.children is None:
#             return False
#         else:
#             return all(c.isTerminal() for c in self.children)
#
#     def allNodes(self):
#         yield self
#         if self.children is not None:
#             for c in self.children:
#                 yield from c.allNodes()
#
#     def depth(self):
#         if self.isTerminal():
#             return 1
#         else:
#
#             return (1 + max(c.depth() for c in self.children))




def backtracking(table, cell):
    """
    table is a table of tuple (rules, coordinates)
    cell is a pair of triple of numbers (i,j,k)
    recursively find the (best) parse tree
    """
    rule, coord = table[cell]

    if coord is None:
        child = tree(rule.right, None)
        t = tree(rule.left, [child])
        return t
    else:
        c1, c2 = coord
        #print (c)
        #l.append((backtracking(table, c1, l), backtracking(table, c2,l)))
        t = tree(rule.left, [backtracking(table, c1), backtracking(table, c2)])

    return t


def backtracking_multiple(table, cell, k_best):
    """
    table is a table of tuple (rules, coordinates)
    cell is a pair of triple of numbers (i,j,k)
    recursively find the (best) parse tree
    """
    first_k = table[cell]

    l = []
    for regola, coord in first_k:

        if coord is None:
            word = regola.right[0]
            rule_ = rule(word, ("__STOP__",))
            child = tree(rule_, None)
            t = tree(regola, [child])
            #t = tree(regola, None)
            l.append(t)
            return l
        else:
            c1, c2 = coord
            #print (c)
            #l.append((backtracking(table, c1, l), backtracking(table, c2,l)))
            #print (len(backtracking_multiple(table, c1, k_best)[:k_best]))
            for child1 in backtracking_multiple(table, c1, k_best):
                for child2 in backtracking_multiple(table, c2, k_best):
                    t = tree(regola, [child1, child2])
                    l.append(t)

    return l[:k_best]


def backtracking_multiple_with_scores(table, cell, k_best):
    """
    table is a table of list of tuples (rules, coordinates)
    coord is a pair of triple of numbers (i,j,k)
    recursively find the k best parse trees (is this a beam search?)
    """
    first_k = table[cell]

    l = []
    for regola, coord in first_k:

        if coord is None:
            word = regola.right[0]
            rule_ = rule(word, ("__STOP__",))
            child = tree(rule_, None)
            t = tree(regola, [child])
            #t = tree(regola, None)
            l.append(t)
            return l
        else:
            c1, c2 = coord

            for child1 in backtracking_multiple_with_scores(table, c1, k_best):
                for child2 in backtracking_multiple_with_scores(table, c2, k_best):
                    t = tree(regola, [child1, child2])
                    l.append(t)

    return sorted(l, key=lambda x: x.score, reverse=True)[:k_best]


def backtracking_multiple_with_scores2(table, cell, k_best, level=1):
    """
    table is a table of list of tuples (rules, coordinates)
    coord is a pair of triple of numbers (i,j,k)
    recursively find the k best parse trees (is this a beam search?)
    """
    #print (level)
    first_k = table[cell][:min(2*level, k_best)]

    l = []
    for regola, coord in first_k:

        if coord is None:
            word = regola.right[0]
            rule_ = rule(word, ("__STOP__",))
            child = tree(rule_, None)
            t = tree(regola, [child])
            #t = tree(regola, None)
            l.append(t)
            return l
        else:
            c1, c2 = coord

            for child1 in backtracking_multiple_with_scores2(table, c1, k_best, level=level+1):
                for child2 in backtracking_multiple_with_scores2(table, c2, k_best, level=level+1):
                    t = tree(regola, [child1, child2])
                    l.append(t)

    return sorted(l, key=lambda x: x.score, reverse=True)[:k_best]



cache = {}
def backtracking_multiple_with_DTK(table, cell, k_best, distributed_vector, dtk_generator=None):
    """
    table is a table of list of tuples (rules, coordinates)
    coord is a pair of triple of numbers (i,j,k)
    recursively find the k best parse trees (is this a beam search?)
    """
    if dtk_generator is None:
        #print ("none")
        dtk_generator = dtk.DT(dimension=4096, operation=operation.random_op)


    def sorting_func(t):
        if t in cache:
            return numpy.dot(cache[t], distributed_vector)
        else:
            cache[t] = dtk_generator.dt(t)
            return numpy.dot(cache[t], distributed_vector)

    def sorting_func_distance_from_1(t):
        if t in cache:
            return 1 / numpy.abs(1 - numpy.dot(cache[t], distributed_vector))
        else:
            tt = dtk_generator.dt(t)
            cache[t] = tt
            return 1 / numpy.abs(1 - numpy.dot(cache[t], distributed_vector))

    #sorting_func = lambda t: numpy.dot(dtk_generator.dt(t) , distributed_vector)

    first_k = table[cell]  #[:k_best]

    l = []
    for regola, coord in first_k:

        if coord is None:
            word = regola.right[0]
            rule_ = rule(word, ("__STOP__",))
            child = tree(rule_, None)
            t = tree(regola, [child])
            #t = tree(regola, None)
            l.append(t)
            #return l
        else:
            c1, c2 = coord

            for child1 in backtracking_multiple_with_DTK(table, c1, k_best, distributed_vector,
                                                         dtk_generator=dtk_generator):
                for child2 in backtracking_multiple_with_DTK(table, c2, k_best, distributed_vector,
                                                             dtk_generator=dtk_generator):
                    t = tree(regola, [child1, child2])
                    #print (t, sorting_func_distance_from_1(t))
                    l.append(t)

    #print (l[0])
    l = sorted(l, key=sorting_func, reverse=True)[:k_best]
    # for i in l:
    #     print(i, sorting_func(i))
    # print("-")
    return l

def top_down_reconstruction(table, cell=None, k_best=5, distributed_vector=None, dtk_generator=None, lista=None):
    """non so se non funziona per bug o il codice è giusto ma concettualmente non puo funzionare"""

    def scorer(t):
        return numpy.dot(dtk_generator.dt(t), distributed_vector)

    if lista is None:
        #se non ho una lista di partenza la costruisco a partire dalla lista dei nodi di partenza
        lista = []
        first = table[cell]
        r,c = first[0]
        rule_ = tree(r.left, None)
        score = scorer(rule_)
        rule_.coord = cell
        #print (t, t.coord)
        lista.append((rule_, score))

    #print ("prima: ", [str(x[0]) for x in lista])
    lista = sorted(lista, key=lambda x: x[1], reverse=True)
    #print ("dopo: ", [str(x[0]) for x in lista])
    #random.shuffle(lista)

    lista_copy = lista[:]
    lista_totale = []
    for (t, score) in lista:
        lista_albero = []   #lista delle possibili espansioni di QUEL'allbero
        #numero_terminali = len(list(t.allTerminalNodes()))

        for index, node in enumerate(t.allTerminalNodes()):
            #print (index, node)

            c1 = node.coord

            if c1 is None:
                continue

            cell = table[c1]
            for elem in cell:
                #print (elem)

                if elem[1] is None:
                    children=[tree(root=elem[0].right[0],children=None)]
                    children[0].coord = None
                    tt = tree(root=node.root, children=children)
                    tt.coord = None
                    ttt = t.add(tt, index)

                    score = scorer(ttt)
                    lista_albero.append((ttt, score))
                    continue
                cell1 = elem[1][0]
                cell2 = elem[1][1]

                #xx = tree(table[cell1][0][0].left)
                xx = tree(elem[0].right[0])
                xx.coord = cell1
                #yy = tree(table[cell2][0][0].left)
                yy = tree(elem[0].right[1])
                yy.coord = cell2
                children=[xx, yy]
                tt = tree(root=node.root, children=children)
                ttt = t.add(tt, index)
                #print (ttt)
                score = scorer(ttt)
                lista_albero.append((ttt, score))
                lista_albero = list(set(lista_albero))


            #lista_albero = sorted(lista_albero, key=lambda x: x[1])

        #print ([str(x[0]) for x in lista_albero])
        lista_totale.extend(lista_albero)


    lista_totale = sorted(lista_totale, key=lambda x: x[1], reverse=True)[:k_best]
    #print ([str(x[0]) for x in lista_totale])
    #random.shuffle(sorted(lista_totale, key=lambda x: x[1]))
    #lista_totale = lista_totale[:k_best]
    if lista_totale:
        #for x in lista_totale:
        #    print (x[0])
        return top_down_reconstruction(table, None, k_best, distributed_vector, dtk_generator, lista_totale)

    else:
        return lista_copy

listone = []
def top_down_reconstruction2(table, cell=None, k_best=5, distributed_vector=None, dtk_generator=None, lista=None):
    #global listone
    """non so se non funziona per bug o il codice è giusto ma concettualmente non puo funzionare"""

    def scorer(t):
        return numpy.dot(dtk_generator.dt(t), distributed_vector)

    if lista is None:
        #se non ho una lista di partenza la costruisco a partire dalla lista dei nodi di partenza
        lista_ = []
        first = table[cell]
        r,c = first[0]
        rule_ = tree(r.left, None)
        score = scorer(rule_)
        rule_.coord = cell
        #print (t, t.coord)
        lista_.append((rule_, score))

    else:
        lista_ = lista
    #print ("prima: ", [str(x[0]) for x in lista])
    lista_ = sorted(lista_, key=lambda x: x[1], reverse=True)
    #print ([(str(x[0]), x[1]) for x in lista_])
    #random.shuffle(lista)

    lista_copy = lista_[:]

    lista_totale = []
    #listone.extend(lista_totale)
    for (t, score) in lista_:
        lista_albero = []   #lista delle possibili espansioni di QUEL'allbero
        #numero_terminali = len(list(t.allTerminalNodes()))

        for index, node in enumerate(t.allTerminalNodes()):
            #print (index, node)

            c1 = node.coord

            if c1 is None:
                continue

            cell = table[c1]
            for elem in cell:
                #print (elem)

                if elem[1] is None:
                    children=[tree(root=elem[0].right[0],children=None)]
                    children[0].coord = None
                    tt = tree(root=node.root, children=children)
                    tt.coord = None
                    ttt = t.add(tt, index)

                    score = scorer(ttt)

                    lista_albero.append((ttt, score))
                    continue
                cell1 = elem[1][0]
                cell2 = elem[1][1]

                #xx = tree(table[cell1][0][0].left)
                xx = tree(elem[0].right[0])
                xx.coord = cell1
                #yy = tree(table[cell2][0][0].left)
                yy = tree(elem[0].right[1])
                yy.coord = cell2
                children=[xx, yy]
                tt = tree(root=node.root, children=children)
                ttt = t.add(tt, index)
                #print (ttt)
                score = scorer(ttt)

                lista_albero.append((ttt, score))
                #lista_albero = list(set(lista_albero))


            #lista_albero = sorted(lista_albero, key=lambda x: x[1])

        #print ([str(x[0]) for x in lista_albero])
        lista_totale.extend(lista_albero)
    lista_totale = list(set(lista_totale))
    #listone.extend(lista_totale)


    lista_totale = sorted(lista_totale, key=lambda x: x[1], reverse=True)[:k_best]
    #print ([str(x[0]) for x in lista_totale])
    #random.shuffle(sorted(lista_totale, key=lambda x: x[1]))
    #lista_totale = lista_totale[:k_best]
    if lista_totale:
        #for x in lista_totale:
        #    print (x[0])
        return top_down_reconstruction2(table, None, k_best, distributed_vector, dtk_generator, lista_totale)

    else:
        return lista_copy

def backtracking_multiple_with_DTK2(table, cell, k_best, distributed_vector, dtk_generator=None, level=1):
    """
    table is a table of list of tuples (rules, coordinates)
    coord is a pair of triple of numbers (i,j,k)
    recursively find the k best parse trees (is this a beam search?)
    """
    if dtk_generator is None:
        #print ("none")
        dtk_generator = dtk.DT(dimension=4096, operation=operation.random_op)


    def sorting_func(t):
        if t in cache:
            return numpy.dot(cache[t], distributed_vector)
        else:
            cache[t] = dtk_generator.dt(t)
            return numpy.dot(cache[t], distributed_vector)

    def sorting_func_distance_from_1(t):
        if t in cache:
            return 1 / numpy.abs(1 - numpy.dot(cache[t], distributed_vector))
        else:
            tt = dtk_generator.dt(t)
            cache[t] = tt
            return 1 / numpy.abs(1 - numpy.dot(cache[t], distributed_vector))

    #sorting_func = lambda t: numpy.dot(dtk_generator.dt(t) , distributed_vector)

    first_k = sorted(table[cell], key=sorting_func, reverse=True)[:min(level, k_best)]

    l = []
    for regola, coord in first_k:

        if coord is None:
            word = regola.right[0]
            rule_ = rule(word, ("__STOP__",))
            child = tree(rule_, None)
            t = tree(regola, [child])
            #t = tree(regola, None)
            l.append(t)
            #return l
        else:
            c1, c2 = coord

            for child1 in backtracking_multiple_with_DTK2(table, c1, k_best, distributed_vector,
                                                         dtk_generator=dtk_generator,level=level+1):
                for child2 in backtracking_multiple_with_DTK2(table, c2, k_best, distributed_vector,
                                                             dtk_generator=dtk_generator,level=level+1):
                    t = tree(regola, [child1, child2])
                    #print (t, sorting_func_distance_from_1(t))
                    l.append(t)

    #print (l[0])
    l = sorted(l, key=sorting_func, reverse=True)[:k_best]
    # for i in l:
    #     print(i, sorting_func(i))
    # print("-")
    return l

def nicerTable(table, grammar):
    n, _, r = table.shape
    M = numpy.zeros((n, n), dtype=list)
    for i in range(n):
        for j in range(n):
            M[i][j] = []

    for i in range(n):
        for j in range(n):
            for k in range(r):
                if table[i][j][k]:
                    M[i][j].append((grammar.nonterminalsymbol[k], table[i][j][k]))

    return M

def nicerTable2(table, grammar):
    #table[i][j][k] contiene una lista di (regole, coordinate), la root della regola è sempre la stessa
    n, _ = table.shape
    M = numpy.zeros((n, n), dtype=list)
    for i in range(n):
        for j in range(n):
            M[i][j] = []

    #se table[i][j][k] -> M[i][j] contiene SIMBOLO k
    for i in range(n):
        for j in range(n):
            
            if table[i][j][k]:
                for elem in table[i,j,k]:
                #print (elem)
                    M[i][j].append(elem)

    return M

def convert_table(table, grammar, mode="3to2"):
    """mode is either "3to2" or "2to3" """
    if mode == "3to2":
        return nicerTable2(table, grammar)
    if mode == "2to3":
        n = table.shape[0]
        r = len(grammar.nonterminalrules)
        P = numpy.zeros((n, n, r), dtype=list)
        for i in range(n):
            for j in range(n):
                for k in range(r):
                    P[i][j][k] = []

        for i in range(n):
            for j in range(n):
                if table[i, j]:
                    for elem in table[i, j]:
                        #print (elem)
                        k = grammar.symbols[elem[0].left]
                        P[i,j,k].append(elem)

        return P

def recognizer(sentence, grammar):
    words = sentence.split()
    n = len(words)
    r = len(grammar.nonterminalsymbol)
    P = numpy.zeros((n, n, r), dtype=bool)
    for i in range(n):
        for rule in grammar.terminalrules:
            #print(rule, i)
            if rule.right[0] == words[i]:
                #print("true: ", rule, i, grammar.nonterminalsymbol.index(rule.left))
                P[i][0][grammar.nonterminalsymbol.index(rule.left)] = True

    for i in range(2, n + 1):
        for j in range(1, n - i + 2):
            for k in range(1, i):
                for rule in grammar.nonterminalrules:
                    a = grammar.nonterminalsymbol.index(rule.left)
                    b = grammar.nonterminalsymbol.index(rule.right[0])
                    c = grammar.nonterminalsymbol.index(rule.right[1])
                    #print(rule, (j-1,k-1,b), " - ", (j+k-1, i-k-1, c))
                    #print (P[j-1][k-1][b], P[j+k-1][i-k-1][c])
                    if P[j - 1][k - 1][b] and P[j + k - 1][i - k - 1][c]:
                        P[j - 1][i - 1][a] = True

    M = nicerTable(P, grammar)

    if any(P[0][-1][:]):
        return P, M, True
    else:
        return P, M, False

def parser(sentence, grammar):
    words = sentence.split()
    n = len(words)
    r = len(grammar.nonterminalsymbol)
    P = numpy.zeros((n, n, r), dtype=object)
    for i in range(n):
        for rule in grammar.terminalrules:
            #print(rule, i)
            if rule.right[0] == words[i]:
                #print("true: ", rule, i, grammar.nonterminalsymbol.index(rule.left))
                index = grammar.nonterminalsymbol.index(rule.left)
                P[i][0][index] = (rule, None)

    for i in range(2, n + 1):
        for j in range(1, n - i + 2):
            for k in range(1, i):
                for rule in grammar.nonterminalrules:
                    a = grammar.nonterminalsymbol.index(rule.left)
                    b = grammar.nonterminalsymbol.index(rule.right[0])
                    c = grammar.nonterminalsymbol.index(rule.right[1])

                    if P[j - 1][k - 1][b] and P[j + k - 1][i - k - 1][c]:
                        P[j - 1][i - 1][a] = (rule, ((j - 1, k - 1, b), (j + k - 1, i - k - 1, c)))


    #find the coordinate of the cell with the final "S"
    fc = None
    for i, v in enumerate(P[0][-1][:]):
        if v:
            fc = i
            break
    if fc:
        final_cell = (0, -1, fc)
    else:
        return False, None

    #must beautify this but it seems to work more or less
    parse = backtracking(P, final_cell)

    if any(P[0][-1][:]):
        return True, parse

def parser_multiple(sentence, grammar, k_best, type="prob", distributed_vector=None, dtk_generator=None):
    words = sentence.split()
    n = len(words)
    r = len(grammar.nonterminalsymbol)

    #initialization of a chart with empty lists
    filler = numpy.frompyfunc(lambda x: list(), 1, 1)
    P = numpy.zeros((n, n, r), dtype=object)
    filler(P, P)

    for i in range(n):
        for rule in grammar.terminalrules:
            #print(rule, i)
            if rule.right[0] == words[i]:
                #print("true: ", rule, i, grammar.nonterminalsymbol.index(rule.left))
                index = grammar.nonterminalsymbol.index(rule.left)
                P[i][0][index].append((rule, None))

    for i in range(2, n + 1):
        for j in range(1, n - i + 2):
            for k in range(1, i):
                for rule in grammar.nonterminalrules:
                    a = grammar.nonterminalsymbol.index(rule.left)
                    b = grammar.nonterminalsymbol.index(rule.right[0])
                    c = grammar.nonterminalsymbol.index(rule.right[1])

                    if P[j - 1][k - 1][b] and P[j + k - 1][i - k - 1][c]:
                        #print ("yeah: ", rule)
                        P[j - 1][i - 1][a].append((rule, ((j - 1, k - 1, b), (j + k - 1, i - k - 1, c))))





    #find the coordinate of the cell with the final "S"
    fc = None
    if P[0][-1][grammar.start_symbol_index]:
        fc = grammar.start_symbol_index
    # for i, v in enumerate(P[0][-1][:]):
    #     print (i,v)
    #     if v:
    #         fc = i

    #M = nicerTable(P, grammar)
    #print (M)

    if fc is not None:
        final_cell = (0, -1, fc)
    else:
        return False, None


    if type == "prob":
        parse = backtracking_multiple_with_scores(P, final_cell, k_best)
    if type == "DTK":
        parse = backtracking_multiple_with_DTK(P, final_cell, k_best, distributed_vector, dtk_generator=dtk_generator)

    #UHMMMMMMMMMMMMMM
    #if parse:
    #if any(P[0][-1][:]):
    return True, parse

def parser_multiple2(sentence, grammar, k_best, type="prob", distributed_vector=None, dtk_generator=None):
    #uso la grammatica nuova e i tree giusti
    #also, uso una seconda matrice B[n,n,r] di backpointers
    words = sentence.split()
    n = len(words)
    r = len(grammar.symbols)

    #initialization of a chart with empty lists
    P = numpy.zeros((n, n, r), dtype=object)
    B = numpy.zeros((n, n, r), dtype=object)
    for i, _ in numpy.ndenumerate(P):
        P[i] = []
        B[i] = []

    #unit production
    for i, word in enumerate(words):
        try:
            rules = grammar.terminalrules[word]
        except KeyError:
            print ("la parola ", word, " non appare nelle regole")

        for rule in rules:
            #print (rule)
            index = grammar.symbols[rule.left]
            P[i][0][index].append((rule, None))
            B[i][0][index].append(None)

    #non terminal rules
    for i in range(2, n + 1):
        for j in range(1, n - i + 2):
            for k in range(1, i):
                for rule in grammar.nonterminalrules:
                    a = grammar.symbols[rule.left]
                    b = grammar.symbols[rule.right[0]]
                    c = grammar.symbols[rule.right[1]]

                    if P[j - 1][k - 1][b] and P[j + k - 1][i - k - 1][c]:
                        P[j - 1][i - 1][a].append((rule, ((j - 1, k - 1, b), (j + k - 1, i - k - 1, c))))
                        B[j - 1][i - 1][a].append(((j - 1, k - 1, b), (j + k - 1, i - k - 1, c)))




    #find the coordinate of the cell with the final "S"
    fc = None
    if P[0][-1][grammar.start_symbol_index]:
        fc = grammar.start_symbol_index

    if fc is not None:
        final_cell = (0, -1, fc)
    else:
        return False, None

    if type == "prob":
        parse = backtracking_multiple_with_scores2(P, final_cell, k_best)
    if type == "DTK":
        #parse = backtracking_multiple_with_DTK(P, final_cell, k_best, distributed_vector, dtk_generator=dtk_generator)
        parse = top_down_reconstruction2(P, final_cell, k_best, distributed_vector,dtk_generator)

    return True, parse

def parser_with_reconstruction(sentence, grammar, k_best, distributed_vector=None, dtk_generator=None, albero=None):
    #uso la grammatica nuova e i tree giusti
    #also, uso una seconda matrice B[n,n,r] di backpointers

    tk = TK(LAMBDA=0.4)

    words = sentence.split()
    n = len(words)
    r = len(grammar.symbols)

    #initialization of a chart with empty lists
    P = numpy.zeros((n, n, r), dtype=object)
    #B = numpy.zeros((n, n, r), dtype=object)
    for i, _ in numpy.ndenumerate(P):
        P[i] = []
        #B[i] = []

    #unit production
    for i, word in enumerate(words):
        try:
            rules = grammar.terminalrules[word]
        except KeyError:
            print ("la parola ", word, " non appare nelle regole")

        for rule in rules:
            #print (rule, rule.toTree())
            rt = rule.toTree()
            index = grammar.symbols[rule.left]
            P[i][0][index].append(((rule, None),(rt, 1)))
            #B[i][0][index].append(None)

        for k in range(r):
            P[i][0][k] = P[i][0][k][:k_best]
            #if len(P[i][0][k]) > 0:
            #    print (len(P[i][0][k]))


    #non terminal rules

    numero_dtk = 0

    for i in range(2, n + 1):
        for j in range(1, n - i + 2):
            for k in range(1, i):
                for rule in grammar.nonterminalrules:
                    a = grammar.symbols[rule.left]
                    b = grammar.symbols[rule.right[0]]
                    c = grammar.symbols[rule.right[1]]

                    if P[j - 1][k - 1][b] and P[j + k - 1][i - k - 1][c]:
                        #rt = rule.toTree()
                        #print (rt)
                        #print (len(P[j - 1][k - 1][b]), " * ", len(P[j + k - 1][i - k - 1][c]))
                        for x, y in itertools.product(P[j - 1][k - 1][b], P[j + k - 1][i - k - 1][c]):

                            subtree1 = x[1][0]
                            subtree2 = y[1][0]
                            rtt = tree(root=rule.left, children=[subtree1, subtree2])
                            #print (rtt)
                            score = numpy.dot(dtk_generator.dt(rtt), distributed_vector)
                            #score = tk.evaluate(rtt, albero)
                            numero_dtk = numero_dtk + 1
                            #P[j - 1][i - 1][a].append(((rule, ((j - 1, k - 1, b), (j + k - 1, i - k - 1, c))), (rtt, score)))
                            P[j - 1][i - 1][a].append(((rule, ((j - 1, k - 1, b), (j + k - 1, i - k - 1, c))), (rtt, score)))

                            #P[j - 1][i - 1].append(((rule, ((j - 1, k - 1, b), (j + k - 1, i - k - 1, c))), (rtt, score)))

                            #print (P[j - 1][k - 1][a])
                        #B[j - 1][i - 1][a].append(((j - 1, k - 1, b), (j + k - 1, i - k - 1, c)))
                        P[j-1][i-1][a] = sorted(P[j-1][i-1][a], key=lambda x: x[1][1], reverse=True)[:k_best]



    print (numero_dtk)



    #find the coordinate of the cell with the final "S"
    fc = None
    if P[0][-1][grammar.start_symbol_index]:
        fc = grammar.start_symbol_index

    if fc is not None:
        return True, P[0][-1][fc]
    else:
        return False, None

def parser_with_reconstruction2(sentence, grammar, k_best, distributed_vector=None, dtk_generator=None):
    #uso la grammatica nuova e i tree giusti
    #also, uso una seconda matrice B[n,n,r] di backpointers

    words = sentence.split()
    n = len(words)
    r = len(grammar.symbols)

    #P = n*[n*[r*[[]]]]

    #initialization of a chart with empty lists
    P = numpy.zeros((n, n), dtype=object)
    for i, _ in numpy.ndenumerate(P):
        P[i] = []
        #B[i] = []

    #unit production
    for i, word in enumerate(words):
        try:
            rules = grammar.terminalrules[word]
        except KeyError:
            print ("la parola ", word, " non appare nelle regole")

        for rule in rules:
            #print (rule, rule.toTree())
            rt = rule.toTree()
            #score = numpy.dot(dtk_generator.dt(rt), distributed_vector)
            P[i][0].append(((rule, None),(rt, 1)))



        P[i][0] = sorted(P[i][0], key=lambda x: x[1][1], reverse=True)




    #non terminal rules

    numero_dtk = 0

    for i in range(2, n + 1):
        for j in range(1, n - i + 2):
            for k in range(1, i):
                for rule in grammar.nonterminalrules:
                    a = grammar.symbols[rule.left]
                    b = grammar.symbols[rule.right[0]]
                    c = grammar.symbols[rule.right[1]]

                    lista_b = [x for x in P[j - 1][k - 1] if x[0][0].left == rule.right[0]]
                    lista_c = [x for x in P[j + k - 1][i - k - 1] if x[0][0].left == rule.right[1]]

                    if lista_b and lista_c:
                        #print (lista_b)
                        #print (lista_c)
                        #rt = rule.toTree()
                        #print (rt)
                        #print (len(P[j - 1][k - 1][b]), " * ", len(P[j + k - 1][i - k - 1][c]))
                        for x, y in itertools.product(lista_b, lista_c):

                            subtree1 = x[1][0]
                            subtree2 = y[1][0]
                            rtt = tree(root=rule.left, children=[subtree1, subtree2])
                            #score = numpy.dot(dtk_generator.sn(rtt), distributed_vector)
                            score = numpy.dot(dtk_generator.dt(rtt), distributed_vector)
                            numero_dtk = numero_dtk + 1
                            #P[j - 1][i - 1][a].append(((rule, ((j - 1, k - 1, b), (j + k - 1, i - k - 1, c))), (rtt, score)))
                            #P[j - 1][i - 1][a].append(((rule, ((j - 1, k - 1, b), (j + k - 1, i - k - 1, c))), (rtt, score)))

                            P[j - 1][i - 1].append(((rule, ((j - 1, k - 1, b), (j + k - 1, i - k - 1, c))), (rtt, score)))
                            #P[j - 1][i - 1] = P[j - 1][i - 1][:k_best]
                            #print (P[j - 1][k - 1][a])
                        #B[j - 1][i - 1][a].append(((j - 1, k - 1, b), (j + k - 1, i - k - 1, c)))

            P[j-1][i-1] = sorted(P[j-1][i-1], key=lambda x: x[1][1], reverse=True)[:k_best]

    print (numero_dtk)




    #print (P[0][-1])

    #for i, l in enumerate(P):
    #    print (i, l)

    #find the coordinate of the cell with the final "S"
    fc = None


    lista_s = [x for x in P[0][-1] if x[0][0].left == "S"]
    if lista_s:
        return True, [t[1][0] for t in lista_s]
    else:
        return False, None

    #
    # if fc is not None:
    #     return True, P[0][-1][fc]
    # else:
    #     return False, None

def parser_with_reconstruction3(sentence, grammar, k_best, distributed_vector=None, dtk_generator=None):
    #uso la grammatica nuova (grammar_2 )


    words = sentence.split()
    n = len(words)
    r = len(grammar.symbols)

    P = numpy.zeros((n, n), dtype=object)
    for i, _ in numpy.ndenumerate(P):
        P[i] = []

    #unit production
    for i, word in enumerate(words):
        try:
            rules = grammar.terminalrules[word]
        except KeyError:
            print ("la parola ", word, " non appare nelle regole")

        if rules == []:
            for symbol in grammar.symbols:
                #print (rule, rule.toTree())
                rule = grammar.rule(symbol,[word])
                rt = rule.toTree()
                score = numpy.dot(dtk_generator.dt(rt), distributed_vector)
                P[i][0].append(((rule, None),(rt, score)))
        else:
            for rule in rules:
                #print (rule, rule.toTree())
                rt = rule.toTree()
                score = numpy.dot(dtk_generator.dt(rt), distributed_vector)
                P[i][0].append(((rule, None),(rt, score)))

        P[i][0] = sorted(P[i][0], key=lambda x: x[1][1], reverse=True)[:k_best]


    #non terminal rules
    numero_dtk = 0

    for i in range(2, n + 1):
        for j in range(1, n - i + 2):
            for k in range(1, i):

                #da qui devo cambiare
                # a = grammar.symbols[rule.left]
                # b = grammar.symbols[rule.right[0]]
                # c = grammar.symbols[rule.right[1]]

                # celle da analizzare, contengono una lista di regole: [   'VP': [@S -> NP VP, VP -> MD VP] , ... ]
                # creo combinazioni di regole con primo simbolo dalla prima cella, secondo dalla seconda
                #
                cella_sinistra = P[j - 1][k - 1]
                cella_destra = P[j + k - 1][i - k - 1]

                stringhe = []
                if cella_sinistra and cella_destra:
                    for x, y in itertools.product(cella_sinistra, cella_destra):
                        #print (x, y)
                        b = x[0][0].left
                        c = y[0][0].left
                        stringhe.append(b + " " + c)
                stringhe = list(set(stringhe))

                if stringhe:
                    pass
                for stringa in stringhe:
                    if rules:
                        pass
                        #print ("rules: ", len(grammar.nonterminalrules[stringa]))
                    for rule in grammar.nonterminalrules[stringa]:

                        subtree1 = cella_sinistra[0][1][0]
                        subtree2 = cella_destra[0][1][0]

                        rtt = tree(root=rule.left, children=[subtree1, subtree2])
                        #print (rtt)
                        #score = numpy.dot(dtk_generator.sn(rtt), distributed_vector)
                        score = numpy.dot(dtk_generator.dt(rtt), distributed_vector)
                        numero_dtk = numero_dtk + 1
                        #P[j - 1][i - 1][a].append(((rule, ((j - 1, k - 1, b), (j + k - 1, i - k - 1, c))), (rtt, score)))
                        #P[j - 1][i - 1][a].append(((rule, ((j - 1, k - 1, b), (j + k - 1, i - k - 1, c))), (rtt, score)))

                        P[j - 1][i - 1].append(((rule, ((j - 1, k - 1, None), (j + k - 1, i - k - 1, None))), (rtt, score)))
                        #P[j - 1][i - 1] = P[j - 1][i - 1][:k_best]
                        #print (P[j - 1][k - 1][a])
                        #B[j - 1][i - 1][a].append(((j - 1, k - 1, b), (j + k - 1, i - k - 1, c)))



            P[j-1][i-1] = sorted(P[j-1][i-1], key=lambda x: x[1][1], reverse=True)[:k_best]

    #print (numero_dtk) #number of iteration

    #print (P[0][-1])

    #for i, l in enumerate(P):
    #    print (i, l)

    #find the coordinate of the cell with the final "S"
    fc = None

    
    lista_s = [x for x in P[0][-1] if x[0][0].left == "S"]

    if lista_s:
        return True, [t[1][0] for t in lista_s], P
    else:
        return False, None

    #
    # if fc is not None:
    #     return True, P[0][-1][fc]
    # else:
    #     return False, None

def parser_jurafsky(sentence, grammar, k_best, distributed_vector=None, dtk_generator=None):

    #COPIARE DAL LIBRO, mettendo come score <dtk, regola>

     #uso la grammatica nuova e i tree giusti
    #also, uso una seconda matrice B[n,n,r] di backpointers
    words = sentence.split()
    n = len(words)
    r = len(grammar.symbols)

    #initialization of a chart with empty lists
    P = numpy.zeros((n, n, r), dtype=float)
    B = numpy.zeros((n, n, r), dtype=tuple)
    for i, _ in numpy.ndenumerate(P):
        P[i] = 0
        B[i] = tuple()

    #unit production
    for i, word in enumerate(words):
        try:
            rules = grammar.terminalrules[word]
        except KeyError:
            print ("la parola ", word, " non appare nelle regole")

        for rule in rules:
            index = grammar.symbols[rule.left]

            P[i][i][index] = rule.prob


    #non terminal rules
    for j in range(2, n + 1):
        for i in range(1, n - j + 2):
            for k in range(1, j):
                for rule in grammar.nonterminalrules:
                    a = grammar.symbols[rule.left]
                    b = grammar.symbols[rule.right[0]]
                    c = grammar.symbols[rule.right[1]]

                    prob = P[i - 1][k - 1][b] * P[i + k - 1][j - k - 1][c] * rule.prob
                    #print (prob)

                    if prob > P[i - 1][j - 1][a]:

                    #if P[j - 1][k - 1][b] and P[j + k - 1][i - k - 1][c]:
                        P[i - 1][j - 1][a] = prob
                        B[i - 1][j - 1][a] = (k, a, b)


    return P, B


def str_diff(s, t):
    return [i for i in range(len(s)) if s[i] != t[i]][0]


def simpleTable(P):
    #new table
    n,_ = P.shape
    M = numpy.zeros((n, n), dtype=object)
    for (i,j), _ in numpy.ndenumerate(M):
        #print (P[i][0][0][0])
        if P[i,j]:
            if len ((P[i,j][0][0][0].right)) == 1:
                M[j,i] = P[i,j][0][0][0].left
            else:
                M[j,i] = P[i,j][0][0][0]
        else:
            M[j,i] = 0

    return M


if __name__ == "__main__":

    #PARAMETER DEFINITION:
    #-grammar:

    #g = grammar(rules)
    #G = grammar.Grammar_(rules)






    distributed = dtk.DT(dimension=1024, LAMBDA=0.4, operation=operation.shuffled_convolution)




    ss = "(S (@S (NP (@NP (@NP (NP (NNP Pierre)(NNP Vinken))(, ,))(ADJP (NP (CD 61)(NNS years))(JJ old)))(, ,))(VP (MD will)(VP (@VP (@VP (VB join)(NP (DT the)(NN board)))(PP (IN as)(NP (@NP (DT a)(JJ nonexecutive))(NN director))))(NP (NNP Nov.)(CD 29)))))(. .))"
    l = tree(string=ss)
    l = tree.binarize(l)
    l = tree.normalize(l)

    sent = tree.sentence(l)
    print (l)
    

    rules = [grammar.fromTreetoRule(x) for x in l.allRules()]
    g = grammar.Grammar(rules)

    _, p, P = parser_with_reconstruction3(sent, g, 1, distributed.dt(l), distributed)



    
    
    T = treeToCYKMatrix.treeToCYKMatrix(l)
    M = simpleTable(P)

    for i, _ in numpy.ndenumerate(M):
        print (M[i], T[i])


    print (all(M.flatten() == T.flatten()))
    

    