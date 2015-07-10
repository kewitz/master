# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:17:40 2015

@author: leo
"""
from numpy import *
import NScheme as ns

m = ns.Mesh(file="./res/teste1_1.msh", verbose=True)
bound = {1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0}
R = 0
c = True

vc, ic, bc = m.run(cuda=False, coloring=False, boundary=bound, errmin=1E-4, kmax=10000, R=R)
print "CPU: %i iterações em %.2f segundos." %(ic, bc)
vg, ig, bg = m.run(cuda='stream', coloring=True, boundary=bound, errmin=1E-4, kmax=10000, R=R)
print "GPUStream: %i iterações em %.2f segundos." %(ig, bg)

# Erro
emax = max([a for a in abs((vg-vc)/vc) if str(a) != 'nan'])
print "Erro relativo máximo: %E" % emax