# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:17:40 2015

@author: leo
"""
from numpy import *
import NScheme as ns

m = ns.Mesh(file="./res/teste1_1.msh", verbose=False)
bound = {1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0}

vc, ic, bc = m.run(cuda=False, boundary=bound, errmin=1E-6, kmax=10000)
print "CPU took %i iterations. MaxV=%E" % (ic, vc.max())
vg, ig, bg = m.run(cuda=True, coloring=True, boundary=bound, errmin=1E-6, kmax=10000)
print "GPU took %i iterations. MaxV=%E" % (ig, vg.max())

err = max([e for e in abs((vc-vg)/vc) if e != nan])
print "Erro relativo máximo: %E" % err
m.plotResult(vc)
