# -*- coding: utf-8 -*-
from numpy import *
import NScheme as ns
import matplotlib.pyplot as plt

path = """./res/"""
plt.style.use('bmh')

alpha = 1E-5
m = ns.Mesh(file=path + "microstrip2.msh", verbose=True)
bound = {1: 0.0, 2: 0.0, 9: 0.0, 10: 0.0, 5: 5.0}

for e in m.elementsByTag([12]):
    e.eps *= 2.9

for n in m.nodesOnLine([1, 2, 9, 10, 5]):
    n.calc = False

cuda = True
v, i, b = m.run(cuda=cuda, boundary=bound, alpha=alpha, R=.1)

m.plotResult(result=v)
