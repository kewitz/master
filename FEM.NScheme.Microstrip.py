# -*- coding: utf-8 -*-
from numpy import *
import NScheme as ns
import matplotlib.pyplot as plt

path = """./res/"""

m = ns.Mesh(file=path + "microstrip.msh", verbose=False)
bound = {1: 0.0, 2: 0.0, 9: 0.0, 10: 0.0, 5: 5.0}

for n in m.nodesOnLine([1, 2, 9, 10, 5]):
    n.calc = False

#vc, ic, bc = m.run(cuda=False, boundary=bound, R=.7, errmin=1E-6, kmax=10000)
#vg, ig, bg = m.run(cuda=True, boundary=bound, R=.6, errmin=1E-6, color=True, kmax=10000)
#print "GPU: %i iterations." % ig
m.coloring(True)

plt.figure(1)
for c in m.colors:
    nodes = matrix([[m.nodes[n].x, m.nodes[n].y] for n in c])
    plt.plot(nodes[:,0], nodes[:,1], 'o')

plt.show()