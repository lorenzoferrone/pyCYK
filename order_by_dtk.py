__author__ = 'lorenzo'


import numpy
import sys
import pandas
import pickle
#sys.path.append("/Users/lorenzo/Documents/Programming/Java/pyDTK2/src")
sys.path.append("/home/ferrone/pyDTK2/src")
import random
#from tree import Tree as tree
import cyk_easy
import dtk
import operation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
from collections import Counter


from grammar import grammar as gram, Rule


def order(grammar, dimension=1024, LAMBDA= 1., max_len=40, depth=6):

    #da mettere fuori dalla funzione

    op = operation.shuffled_convolution
    distributed = dtk.DT(dimension=dimension, LAMBDA=LAMBDA, operation=op)

    #random.seed(10)
    start_symbol = "S"  #modificare per le altre grammatiche

    lista = []


    #va bene se tengo questa formulazione pure per le altre grammatiche?
    s = grammar.parsableString(start_symbol=start_symbol, depth=depth)
    frase = s.sentence
    #print (frase)
    l = len(frase.split(" "))

    if l <= max_len:
        _, p = cyk_easy.parser_multiple(s.sentence, g, 1000)
        #print (len(p))
        if len(p) > 10:     #only save when there are more then 10 choices
            t = random.choice(p)


    #print (t)
    try:
        dt = distributed.dt(t)
        for f in p:
            #print (f)
            df = distributed.dt(f)
            k = numpy.dot(df, dt)
            lista.append([f, k])
        return lista, t
    except UnboundLocalError:
        #print ("retry")
        return [], None






if __name__ == '__main__':
    rules = [Rule("S", ("NP", "VP"), prob=0.8),
             Rule("S", ("X1", "VP"), prob=0.15),
             Rule("S", ("X2", "PP")),
             Rule("S", ("V", "NP")),
             Rule("S", ("V", "PP")),
             Rule("S", ("VP", "PP")),
             Rule("S", ("VP",)),
             Rule("S", ("book",)),
             Rule("S", ("include",)),
             Rule("S", ("prefer",)),

             Rule("NP", ("DET", "NOM",), prob=0.2),
             Rule("NP", ("TWA",)),
             Rule("NP", ("HOUSTON",)),
             Rule("NP", ("me",)),
             Rule("NP", ("she",)),
             Rule("NP", ("I",)),

             Rule("NOM", ("NOM", "N")),
             Rule("NOM", ("NOM", "PP")),
             Rule("NOM", ("book",)),
             Rule("NOM", ("flight",)),
             Rule("NOM", ("meal",)),
             Rule("NOM", ("money",)),
             Rule("NOM", ("morning",)),

             Rule("X1", ("AUX", "NP"), prob=0.15),
             Rule("X2", ("V", "NP")),

             Rule("VP", ("V", "NP"), prob=0.4),
             Rule("VP", ("X2", "PP")),
             Rule("VP", ("V", "PP")),
             Rule("VP", ("VP", "PP")),

             Rule("VP", ("book",)),
             Rule("VP", ("include",)),
             Rule("VP", ("prefer",)),

             Rule("PP", ("P", "NP")),

             Rule("DET", ("that",), prob=0.05),
             Rule("DET", ("this",), prob=0.05),
             Rule("DET", ("a",), prob=0.15),
             Rule("DET", ("the",), prob=0.8),

             Rule("N", ("book",)),
             Rule("N", ("flight",)),
             Rule("N", ("meal",)),
             Rule("N", ("money",)),
             Rule("N", ("fish",)),
             Rule("N", ("fork",)),
             Rule("N", ("morning",)),

             Rule("V", ("book",)),
             Rule("V", ("include",)),
             Rule("V", ("prefer",)),
             Rule("V", ("fish",)),

             Rule("PRO", ("me",)),
             Rule("PRO", ("she",)),
             Rule("PRO", ("I",)),

             Rule("PR-N", ("TWA",)),
             Rule("PR-N", ("HOUSTON",)),

             Rule("AUX", ("does",)),

             Rule("P", ("from",)),
             Rule("P", ("to",)),
             Rule("P", ("on",)),
             Rule("P", ("near",)),
             Rule("P", ("through",)),
    ]



    df = pickle.load(open("dataset_examples1000_len40_depth6", "rb"))
    print (df["sentence","tree"])



    #PARAMETER DEFINITION:
    #-grammar:
    # g = gram(rules)
    # dimension = 2048
    # LAMBDA = 0.4

    # tentativi = 1000


    # for dimension in [1024, 2048, 4096]:
    #     for LAMBDA in [0.2, 0.4, 0.6, 0.8, 1.]:

    #         N = 0

    #         print ("dimension: ", dimension, "lambda: ", LAMBDA)
    #         for i in range(tentativi):
    #             l, t = order(g, dimension=dimension, LAMBDA=LAMBDA)
    #             if l:
    #                 l = sorted(l,key=lambda x: x[1], reverse=True)

    #                 #print (t)
    #                 for i, alb in enumerate(l):
    #                     if alb[0] == t:
    #                         #print (alb[0])
    #                         print (i, end=" ")


    #                 N = N + 1

    #         print ("\n", N)


