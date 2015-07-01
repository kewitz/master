# -*- coding: utf-8 -*-
from numpy import *
import NScheme as ns
import matplotlib.pyplot as plt

path = """./res/"""

m = ns.Mesh(file=path + "microstrip.msh", verbose=False)
bound = {1: 0.0, 2: 0.0, 9: 0.0, 10: 0.0, 5: 5.0}

vc, ic, bc = m.run(cuda=False, boundary=bound, errmin=1E-6, kmax=10000)