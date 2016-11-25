__author__ = 'lorenzo'

import sys
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("/Users/lorenzo/Documents/Programming/Java/pyDTK2/src")
from tree import Tree as tree

f = open("/Users/lorenzo/Documents/Universita/PHD/Lavori/Codice/pyCYK/readTreeresults22_1024")


total = 0
parsed = 0
giusti = 0
dict_frasi = defaultdict(int)
#mettere un modo per contare quante sono per ogni lunghezza
counter_lungh = defaultdict(int)
for line in f.readlines():

    if line.startswith("parsing"):
        total = total + 1

        lungh = len(line.split(" ")[2:])
        counter_lungh[lungh] = counter_lungh[lungh] + 1
    if line.startswith("OK"):
        parsed = parsed + 1
    if line.startswith("True"):
        giusti = giusti + 1
        dict_frasi[lungh] = dict_frasi[lungh] + 1


#print (dict_frasi)
l = list(dict_frasi.items())
print (l)
ll = list(zip(*l))
#print (ll)
plt.plot(*ll)
plt.show()


m = list(counter_lungh.items())
print (m)
mm = list(zip(*m))
#print (mm)
plt.plot(*mm)
plt.show()



percs = []
for (lungh, corretti),( _, tot) in zip(l,m):
    perc = 100*corretti/tot
    percs.append(perc)
    print (lungh, " - ", perc, "su", tot)


print ([x[0] for x in l])
plt.plot([x[0] for x in l], percs)
plt.show()

print (total, parsed, giusti)

