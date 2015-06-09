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

for cuda in [False, True, 'stream']:
    del m
    m = ns.Mesh(file=path + "microstrip2.msh", verbose=False)
    bound = {1: 0.0, 2: 0.0, 9: 0.0, 10: 0.0, 5: 5.0}
    
    for n in m.nodesOnLine([1, 2, 9, 10, 5]):
        n.calc = False

    v, i, b = m.run(cuda=cuda, boundary=bound, R=0, errmin=1E-6)
    print "CUDA %s : %fs" % (cuda, b[0])
    