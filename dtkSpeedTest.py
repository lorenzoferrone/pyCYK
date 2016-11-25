import numpy
import sys

sys.path.append("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/PyDTK/src")
import time
import itertools

import dtk
import dtk2
import operation as op
from tree import Tree as tree

from loadPennTree import loadPennTree

from multiprocessing import Pool


from line_profiler import LineProfiler

def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    return inner


if __name__ == '__main__':
    PennTreePathBinarized = "/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB2/"

    sections = ['21', '23', '24']
    N = 100
    # takes first N element from iterator
    treeIterator = loadPennTree(PennTreePathBinarized, sections, normalize=True)
    # treeList = list(treeIterator)[:100]
    treeIterator = itertools.islice(treeIterator, 0, N)
    treeList= list(treeIterator)

    # print (treeList)

    dtk_generator = dtk.DT(dimension=8192, LAMBDA=0.6, operation=op.fast_shuffled_convolution)
    dtk_generator2 = dtk2.DT(dimension=8192, LAMBDA=0.6, operation=op.fast_shuffled_convolution)
    dtk_generator3 = dtk2.partialTreeKernel(dimension=8192, LAMBDA=0.6, operation=op.fast_shuffled_convolution)

    @do_profile(follow=[dtk_generator2.dt, dtk_generator2.sRecursive])
    def f():
        for t in treeList:
            v = dtk_generator2.dt(t)


    # f()
    print ('vectorizing: ')


    #
    #
    # p = Pool(4)
    # treeList = [tree(string='(A (B C))')]
    start = time.time()

    for t in treeList:
        v = dtk_generator2.dt(t)
        print (v)

    #
    #
    #
    # p.map(dtk_generator.dt, treeList)
    #
    time_ = time.time() - start
    print (N, "alberi in", time_, "secondi. Media:", time_/N, "tree/second:", N/time_)
