# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:46:00 2015

@author: leo
"""

from numpy import *
import NScheme as ns
import matplotlib.pyplot as plt

path = """./res/"""
split = ns.split
m = 1

print "DOF\tCUDA\tTime\t\tIterations\tErr."
for c in [(c, f) for f in ['teste1_1.msh', 'teste1_2.msh', 'teste1_3.msh', 'teste1_4.msh', 'teste1_5.msh'] for c in [False, True]]:
    del m
    cuda, fi = c
    m = ns.Mesh(file=path + fi, verbose=False)
    bound = {1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0}
    
    for n in m.nodesOnLine([1, 2, 3, 4]):
        n.calc = False
    
    if cuda:
        vc = v

    v, i, b = m.run(cuda=cuda, boundary=bound, R=0, errmin=1E-6, kmax=1000000)

    e = 0.0
    if cuda:
        e = max(abs(vc - v))

    print "%i\t%s\t%fs\t%i\t%f" % (len(m.nodes), cuda, b[0], i, e)