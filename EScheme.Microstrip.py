# -*- coding: utf-8 -*-
from numpy import *
import EScheme as es
import matplotlib.pyplot as plt

path = """./res/"""

m = es.Mesh(file=path + "microstrip2.msh", verbose=True)
bound = {1: 0.0, 2: 0.0, 9: 0.0, 10: 0.0, 5: 5.0}

for n in m.nodesOnLine([1, 2, 9, 10, 5]):
    n.calc = False

for i in range(10):
    vc, ic, bc = m.run(cuda=False, boundary=bound, errmin=1E-6, maxiter=1000)
#    print ic
#vg, ig, bg = m.run(cuda=True, boundary=bound, R=0, errmin=1E-6, kmax=10000)
m.plotResult(result=vc)