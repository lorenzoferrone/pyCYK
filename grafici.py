__author__ = 'lorenzo'

import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle



def fscore(prec, rec):
    return 2*prec*rec/(prec+rec)


prec = 0.99
rec = 0.96

print (fscore(prec, rec))

sns.set_context("paper")
sns.set_style({"axes.linewidth" : 1,'legend.frameon': True})
sns.set(font_scale=1.6)

lista_corretti_15 = [22.26, 48.8, 77.9, 91.86, 92.59]
lista_corretti_2 = [23.5, 60.46, 81.39, 88.28, 92.54]
lista_corretti_25 = [23.25, 52.32, 75.58, 87.5, 92.79]

#plt.figure(figsize=(18, 12), dpi=400)

# for label, l in zip(["p = 1.5", "p = 2", "p = 2.5"], [lista_corretti_15, lista_corretti_2, lista_corretti_25]):
#     plt.plot([1024, 2048, 4096, 8192, 16384], l, label=label)
#
# plt.legend(loc=4)
# plt.ylabel("Exactly Reconstructed (%)")
# plt.xlabel("Distributed Trees Dimension")
# #plt.savefig("/Users/lorenzo/Desktop/Current/EMNLP2015/corretti.png", dpi = 200,figsize=(800, 600) )
# plt.show()


lista_precision_15 = [0.89, 0.964, 0.984, 0.994, 0.995]
lista_recall_15 = [0.285, 0.58, 0.846, 0.959, 0.965]
lista_fscore_15 = [fscore(x, y) for x, y in zip(lista_precision_15, lista_recall_15)]



lista_precision_2 = [0.78, 0.912, 0.967, 0.994, 0.995]
lista_recall_2 = [0.43, 0.754, 0.923, 0.959, 0.965]
lista_fscore_2 = [fscore(x, y) for x, y in zip(lista_precision_2, lista_recall_2)]

lista_precision_25 = [0.71, 0.85, 0.951, 0.99, 0.994]
lista_recall_25 = [0.477, 0.78, 0.929, 0.967, 0.976]
lista_fscore_25 = [fscore(x, y) for x, y in zip(lista_precision_25, lista_recall_25)]

lines = ["-","--",":"]
lines = iter(lines)
for label, l in zip(["p = 1.5", "p = 2", "p = 2.5"], [lista_fscore_15, lista_fscore_2, lista_fscore_25]):
    plt.plot([1024, 2048, 4096, 8192, 16384], l, label=label, linestyle=next(iter(lines)))

plt.legend(loc=4)
plt.ylabel("F-measure")
plt.xlabel("Distributed Trees Dimension")
plt.savefig("/Users/lorenzo/Desktop/fscore.png", dpi = 200,figsize=(800, 600) )
plt.show()

# for label, l in zip(["Precision", "Recall", "F-measure"], [lista_precision_15, lista_recall_15, lista_fscore_15]):
#     plt.plot([1024, 2048, 4096, 8192, 16384], l, label=label)
#
# plt.legend(loc=4)
# #plt.ylabel("")
# plt.xlabel("Distributed Trees Dimension")
# plt.savefig("/Users/lorenzo/Desktop/Current/EMNLP2015/prec_rec_f.png", dpi = 200,figsize=(800, 600) )
# plt.show()