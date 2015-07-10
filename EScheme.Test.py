# -*- coding: utf-8 -*-
from numpy import *
import EScheme as es
import matplotlib.pyplot as plt

path = """./res/"""

m = es.Mesh(file=path + "teste1_4.msh", verbose=True)
bound = {1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0}
c = True

vc, ic, bc = m.run(cuda=False, coloring=c, boundary=bound, errmin=1E-4, kmax=10000)
print "CPU: %i iterações em %.2f segundos." %(ic, bc)
vg, ig, bg = m.run(cuda=True, coloring=c, boundary=bound, errmin=1E-4, kmax=10000)
print "GPU: %i iterações em %.2f segundos." %(ig, bg)

emax = max([a for a in abs((vg-vc)/vc) if str(a) != 'nan'])
print "Erro relativo máximo: %E" % emax