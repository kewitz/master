# -*- coding: utf-8 -*-
from numpy import *
import EScheme as es
import matplotlib.pyplot as plt

path = """./res/"""

m = es.Mesh(file=path + "microstrip2.msh", verbose=True)
bound = {1: 0.0, 2: 0.0, 9: 0.0, 10: 0.0, 5: 5.0}

for e in m.elementsByTag([12]):
    e.eps *= 2.9

for n in m.nodesOnLine([1, 2, 9, 10, 5]):
    n.calc = False

errmin = 1E-7

v, ic, bc = m.run(cuda=False, boundary=bound, errmin=errmin)
print "Took %i iterations." % (ic)

m.plotResult(result=v)