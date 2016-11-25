__author__ = 'lorenzo'


import numpy
import sys
import pandas
import pickle
#sys.path.append("/Users/lorenzo/Documents/Programming/Java/pyDTK2/src")
sys.path.append("/home/ferrone/pyDTK2/src")
import random
from tree import Tree as tree
import cyk_easy
import dtk
import operation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
from collections import Counter


from grammar import grammar as gram, Rule


def create_dataset(grammar, num_examples, max_len=40, depth=5):
    #return a pandas.dataframe

    #fare in modo che tutte le lunghezze siano ugualmente rappresentate.... ?

    random.seed(10)
    start_symbol = "S"  #modificare per le altre grammatiche

    rows = []

    length_count = Counter()
    for i in range(num_examples):
        #va bene se tengo questa formulazione pure per le altre grammatiche?
        s = grammar.parsableString(start_symbol=start_symbol, depth=depth)
        frase = s.sentence
        l = len(frase.split(" "))

        if l <= max_len:
            _, p = cyk_easy.parser_multiple(s.sentence, g, 1000)
            if len(p) > 10:     #only save when there are more then 10 choices
                t = random.choice(p)
                rows.append({"sent": frase, "tree": t, "len": l, "choices": len(p)})

        length_count[l] += 1 #devo vedere cosa farci

        #     else:
        #         continue
        #
        # else:
        #     continue

    df = pandas.DataFrame(rows)

    return df

def load_dataset(grammar, num_examples, max_len=40, depth=5):
    #try to load dataset, otherwise it creates it and then dumps it to file

    #TODO: add something to file_name to identify the grammar we are using
    file_name = "dataset" + "_examples" + str(num_examples) + "_len" + str(max_len) + "_depth" + str(depth)
    try:
        dataset = pickle.load(open(file_name, "rb"))
        print ("Dataset loaded")
    except IOError as e:
        print (e)
        print ("Dataset not present, creating one... ", end="")
        dataset = create_dataset(grammar, num_examples=num_examples, max_len=max_len,depth=depth)
        pickle.dump(dataset, open(file_name, "wb"))
        print ("Done!")

    #param = {"grammar": grammar, "examples": num_examples, "len":max_len,"depth":depth}
    #dataset._metadata = param
    #print (dataset["tree"])
    return dataset

def run_experiment(df, dtk, grammar,RANK=1):

    #lista = []
    l_col = []
    for i, row in df.iterrows():
        if i % 100 == 0:
            print (i)
            dtk.cache = {}
            cyk_easy.cache = {}
        print (".", end="")

        frase = row["sent"]
        l = row["len"]
        t = row["tree"]
        #print (t)
        dt = dtk.dt(t)

        _, reconstruct = cyk_easy.parser_multiple(frase, grammar, l, type="DTK", distributed_vector=dt, dtk_generator=dtk)


   
        if RANK == 0:
            l_col.append(reconstruct[0] == t) #OLD VERSION, controlla solo il primo
            #print (reconstruct[0])

        else:
        #new version, chech if the tree is in the first RANK (rank=1 is the same as before)
            trovato = False
            for i in range(RANK):
                if reconstruct[i] == t:
                    trovato = True
                    break
            l_col.append(trovato)


    df["correct"] = l_col
    #df.meta = meta
    return df

def clip(df, clip_len):
    #return the dataset with every length sentence equally-represented
    pass


def stats(df):
    #compute some stats on the dataset df

    #by legth of sentence
    l = []

    m = min(df["len"])
    M = max(df["len"])
    for i in range(m,M):
        d = (df[df["len"]==i])
        if not d.empty:
            num_corretti = d[d["correct"] == True].shape[0]
            num_totali = d.shape[0]
            ratio = num_corretti/num_totali
            #print (i, num_corretti, num_totali, ratio)



            #ratio = df[df["correct"] == True].shape[0] / df.shape[0]
            l.append((i, ratio, num_totali))

        else:
            l.append((i, float("NaN"), 0))

    return l



if __name__ == "__main__":
    rules = [Rule("S", ("NP", "VP")),
             Rule("S", ("X1", "VP")),
             Rule("S", ("X2", "PP")),
             Rule("S", ("V", "NP")),
             Rule("S", ("V", "PP")),
             Rule("S", ("VP", "PP")),
             Rule("S", ("VP",)),
             Rule("S", ("book",)),
             Rule("S", ("include",)),
             Rule("S", ("prefer",)),

             Rule("NP", ("DET", "NOM",)),
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

             Rule("X1", ("AUX", "NP")),
             Rule("X2", ("V", "NP")),

             Rule("VP", ("V", "NP")),
             Rule("VP", ("X2", "PP")),
             Rule("VP", ("V", "PP")),
             Rule("VP", ("VP", "PP")),

             Rule("VP", ("book",)),
             Rule("VP", ("include",)),
             Rule("VP", ("prefer",)),

             Rule("PP", ("P", "NP")),

             Rule("DET", ("that",)),
             Rule("DET", ("this",)),
             Rule("DET", ("a",)),
             Rule("DET", ("the",)),

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




    #PARAMETER DEFINITION:
    #-grammar:
    g = gram(rules)
    num_examples = 10000    #non ha senso che sia un parametro che mi salvo nel nome del file.....
    max_len = 40
    depth = 6
    #-dtk:
    #dimension = 1024
    #LAMBDA = 1.
    op = operation.shuffled_convolution
    #
    rank = 1

    for dimension in [1024, 2048, 4096]:
        for LAMBDA in [0., 0.2, 0.4, 0.6, 0.8, 1.]:


            #parameter_string
            param = "_____dim=" + str(dimension) + "_lambda=" + str(LAMBDA) + "_rank=" + str(rank)

            print (param)


            dataset = load_dataset(g, num_examples=num_examples, max_len=max_len,depth=depth)
            #dtk initialization
            distributed = dtk.DT(dimension=dimension, LAMBDA=LAMBDA, operation=op)

            #run experiment and dump results
            df = run_experiment(dataset, distributed, g, RANK=rank)

            
            #provo a commentare questa parte e usare direttamente df rinominato in results
            #pickle.dump(df, open("results" + param, "wb"))
            #results = pickle.load(open("results" + param, "rb"))
            results = df

            l = stats(results) #l is list of (length, ratio of correct), c is list of (length, total choices for that length)


            #PLOTTING
            
            x_coords = [i[0] for i in l]
            ratios = [i[1] for i in l]
            choices = [i[2] for i in l]
            points = zip(x_coords, ratios)

            #ESTHETIC STUFF
            ax = plt.subplot(111)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(True)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(True)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            
            plt.ylim(0., 1.1)
            m = min(x_coords) - 1
            M = max(x_coords) + 1
            plt.xlim(m, M)
            
            for y in numpy.arange(0., 1.5, 0.1):
               plt.plot(range(m, M+1), [y]*len(range(m, M+1)), "--", lw=0.5, color="black", alpha=0.3)
            
            
            plt.tick_params(axis="both", which="both", bottom="off", top="off",
                        labelbottom="on", left="off", right="off", labelleft="on")

            plt.xlabel("sentence length")
            plt.ylabel("percentage of correctly classified among the first " + str(rank))

            #plt.text(m+1, 0.1, "rank = " + str(rank), fontsize=14, verticalalignment='top', alpha=0.5)
            
            #the plot itself
            plt.scatter(x_coords, ratios)
            for txt, coord in zip(choices,points):
                ax.annotate(txt, xy=coord)
            
            plt.savefig("parse_graph/CorrectVersion_with_seed_2shuffle_" + param + ".png")
            #plt.show()

