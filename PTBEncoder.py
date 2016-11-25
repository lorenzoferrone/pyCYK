import sys
sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
#sys.path.append("/home/ferrone/pyDTK2/src")
import dtk2 as dtk
import sentence_encoder
import loadPennTree
import operation as op
import numpy

dtk_generator = dtk.DT(dimension=8192, LAMBDA=0.4, operation=op.fast_shuffled_convolution)

if __name__ == "__main__":
    path = "/Users/lorenzo/Desktop/Current/Greg/dtknn/PTB"
    sections = "00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24".split()
    l = list(loadPennTree.loadPennTree(path, sections[2:4], True))
    print (len(l))
    L = []
    for i, t in enumerate(l):
        if i%100 == 0:
            print (".", end='', flush=True)
        if i%1000 == 0:
            print (i, flush=True)
        sentence = t.taggedSentence
        v = sentence_encoder.encoder(sentence, dtk_generator, 3)
        L.append([v])
        dtk_generator.cleanCache()

    M = numpy.concatenate(L)
    numpy.save("/Users/lorenzo/Desktop/Current/Greg/dtknn/encoded/train.npy", M)
