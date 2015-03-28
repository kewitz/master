# -*- coding: utf-8 -*-
from numpy import *
import NScheme as ns
import matplotlib.pyplot as plt

path = """./res/"""

alpha = 1E-5
m = ns.Mesh(file=path + "teste1_1.msh", verbose=False)
bound = {1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0}

for n in m.nodesOnLine([1, 2, 3, 4]):
    n.calc = False

cuda = True
vc, ic, bc = m.run(cuda=cuda, boundary=bound, alpha=alpha, R=0)
print "R=0 converged in %i iterations." % ic

for r in arange(.1,1,.1):
    v, i, b = m.run(cuda=cuda, boundary=bound, alpha=alpha, R=r)
    e = abs(vc - v).max()
    print "R=%f converged in %i iterations. Emax=%f" % (r, i, e)

m.plotResult(result=v)