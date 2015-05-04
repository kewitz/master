# -*- coding: utf-8 -*-
from numpy import *
import NScheme as ns
import matplotlib.pyplot as plt

path = """./res/"""

m = ns.Mesh(file=path + "microstrip2.msh", verbose=True)
bound = {1: 0.0, 2: 0.0, 9: 0.0, 10: 0.0, 5: 5.0}

for e in m.elementsByTag([12]):
    e.eps *= 2.9

for n in m.nodesOnLine([1, 2, 9, 10, 5]):
    n.calc = False

alpha = 1E-5
R = 0

vc, ic, bc = m.run(cuda=False, boundary=bound, alpha=alpha, R=R)
vg, ig, bg = m.run(cuda=False, boundary=bound, alpha=alpha, R=R)

m.plotResult(result=vc-vg)
