# -*- coding: utf-8 -*-
from numpy import *
import EScheme as es
import matplotlib.pyplot as plt

path = """./res/"""

m = es.Mesh(file=path + "L10.msh", verbose=True)
bound = {2: 100.0, 5: 0.0}
c = False

vc, ic, bc = m.run(cuda=False, coloring=c, boundary=bound, errmin=1E-4, kmax=10000)
print "CPU: %i iterações em %.2f segundos." % (ic, bc)
vg, ig, bg = m.run(cuda=True, coloring=c, boundary=bound, errmin=1E-4, kmax=10000)
print "GPU: %i iterações em %.2f segundos." % (ig, bg)

emax = max([a for a in abs((vg-vc)/vc) if str(a) != 'nan'])
print "Erro relativo máximo: %E" % emax
m.plotResult(vg)