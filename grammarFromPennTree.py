__author__ = 'lorenzo'

import os
import grammar
import sys
sys.path.append("/Users/lorenzo/Documents/Programming/Java/pyDTK2/src")
import codecs
import pickle


from tree import Tree as tree



folder = "/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB2/"

sezioni = "00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24".split()
sezioni_training = sezioni[:23]
sezioni_developing = sezioni[23:24]
sezioni_testing = sezioni[24:]


rules = []
for sezione in sezioni_training:
    #print (sezione, end= "")

    subfolder = folder + sezione + "/"
    print (subfolder)
    for file_ in os.listdir(subfolder):
        print (file_)
        if file_ in [".DS_Store", ".DS_Storebinarized.txt"]:
            continue
        #print (file_)
        f = codecs.open(subfolder + file_, encoding="utf-8")
        for line in f:
            #print (line[:-1])
            try:
                t = tree(string=line[:-1])
                sent = t.sentence
                if "=" in sent:
                    print (sent)
                    continue
                #devo mettere l'altro livello di "binarizzazione"
                t.binarize()
                t.normalize()
            except Exception as e:
                print (file_, line)
                print (e.with_traceback())
            rules.extend([grammar.fromTreetoRule(x) for x in t.allRules()])

print ("")

G = grammar.Grammar(rules)

print (len(G.nonterminalrules))
print (len(G.terminalrules))
print (len(G.symbols))




pickle.dump(G, codecs.open("grammarPennTree5.txt", "wb"))
