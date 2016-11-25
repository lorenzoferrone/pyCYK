__author__ = 'lorenzo'

import sys
if sys.platform == 'darwin': # sto sul mio pc
    sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
else: #sto deployando su venere
    sys.path.append("/home/ferrone/pyDTK_sync2/src")
import os
from tree import Tree
import numpy

def loadPennTree(pathToTreeBank, sections, normalize=True):
    # returns a list of trees

    if os.path.isfile(pathToTreeBank):
        file = open(pathToTreeBank)
        for line in file:
            tree = Tree(string = line)
            if normalize:
                tree.binarize() # <- problema! questa serve solo con gli alberi binarizzati...
            tree.normalize() #remove case
            yield tree

    #if the path is a directory
    else:
        for (subPath, _, files) in os.walk(pathToTreeBank):
            _.sort()
            files.sort()  #needs to sort it cause os.walk is not consistent across platform
            currentSection = subPath[-2:]
            if currentSection in sections:
                for file in files:
                    if file.startswith('wsj'): #skip other files in the directory
                        print ("processing file: ", file)
                        file = open(os.path.join(subPath, file))
                        for line in file:
                            tree = Tree(string = line)
                            if normalize:
                                tree.binarize()
                            tree.normalize() #remove case
                            yield tree


def distributedMatrix(dtkGenerator, trees):
    dimension = dtkGenerator.dimension
    matrix = numpy.empty(shape=(0, dimension)) #create empty matrix to which append distributed vectors

    for tree in trees:
        distributedTree = dtkGenerator.dt(tree).reshape(1, dimension) #from a d-vector to a (d x 1)-matrix
        matrix = numpy.vstack((matrix, distributedTree))               #append the vector to the matrix

    return matrix


if __name__ == "__main__":
    PennTreePath = "/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB2/"
    DeUnarizedPTB = "/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB2_deunarized/"
