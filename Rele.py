# -*- coding: utf-8 -*-
from numpy import *
import NScheme as solver

path = """./res/"""
mi0, eps0 = (4*pi*1E-7), 8.854187817E-12

m = solver.Mesh(file=path + "rele.msh", verbose=False)
bound = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}

for e in m.elements:
    e.mat = 1./(1.0*mi0)

for e in m.elementsByTag([24, 26]):
    e.mat = 1./(1000*mi0)

for e in m.elementsByTag([22]):
    e.f = 2000000

vc, ic, bc = m.run(cuda=True, boundary=bound, errmin=1E-4, kmax=4000)
print "Took %i iterations." % (ic)

m.plotResult(result=vc)
