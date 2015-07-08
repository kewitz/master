# -*- coding: utf-8 -*-
from numpy import *
from ctypes import *
import EScheme as solver

path = """./res/"""

m = solver.Mesh(file=path + "rele.msh", verbose=True)
bound = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}

for n in m.nodesOnLine(bound.keys()):
    n.calc = False

elements = [e for e in m.elements if e.dim is 2]
colors = []
c = 0
while len(elements) > 0:
    colors.append([])
    colors[c] = [elements.pop()]
    nodes = [n for e in colors[c] for n in e.nodes]
    for i, e in enumerate(elements):
        if len(set(e.nodes).intersection(nodes)) is 0:
            colors[c].append(elements.pop(i))
            nodes = [n for e in colors[c] for n in e.nodes]
    print c, len(colors[c])
    c += 1
