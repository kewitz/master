# -*- coding: utf-8 -*-
from numpy import *
import EScheme as es
import matplotlib.pyplot as plt

path = """./res/"""

m = es.Mesh(file=path + "microstrip2.msh", verbose=True)
bound = {1: 0.0, 2: 0.0, 9: 0.0, 10: 0.0, 5: 5.0}

vg, ig, bg = m.run(cuda=True, boundary=bound, errmin=1E-4, kmax=10000)
m.plotResult(result=vg)