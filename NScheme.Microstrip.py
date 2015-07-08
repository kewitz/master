# -*- coding: utf-8 -*-
from numpy import *
import NScheme as ns
import matplotlib.pyplot as plt

path = """./res/"""

m = ns.Mesh(file=path + "microstrip2.msh", verbose=False)
bound = {1: 0.0, 2: 0.0, 9: 0.0, 10: 0.0, 5: 5.0}
R = 0.74

vc, ic, bc = m.run(cuda=False,coloring=True, boundary=bound, errmin=1E-5, kmax=10000, R=R)
print "CPU: %i iterações em %.2f segundos." %(ic, bc[0])
vg, ig, bg = m.run(cuda=True, coloring=True, boundary=bound, errmin=1E-5, kmax=10000, R=R)
print "GPU: %i iterações em %.2f segundos." %(ig, bg[0])

# Erro
emax = max([a for a in abs((vg-vc)/vc) if str(a) != 'nan'])
print "Erro relativo máximo: %E" % emax