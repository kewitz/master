# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:17:40 2015

@author: leo
"""
from numpy import *
import NScheme as ns

m = ns.Mesh(file="./res/teste1_1.msh", verbose=True)
bound = {1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0}
vc, ic, bc = m.run(cuda=False, boundary=bound, errmin=1E-6, kmax=10000)

m.plotResult(vc)