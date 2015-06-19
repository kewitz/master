# -*- coding: utf-8 -*-
from numpy import *
import EScheme as es
import matplotlib.pyplot as plt

path = """./res/"""

m = es.Mesh(file=path + "teste1_1.msh", verbose=True)
bound = {1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0}

for n in m.nodesOnLine([1, 2, 3, 4]):
    n.calc = False

vc, ic, bc = m.run(cuda=True, boundary=bound, errmin=1E-8, maxiter=1000)
print "Took %i iterations." % (ic)

m.plotResult(result=vc)