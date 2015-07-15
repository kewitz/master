# -*- coding: utf-8 -*-
from numpy import *
from ctypes import *
import NScheme as solver

path = """./res/"""

m = solver.Mesh(file=path + "teste1_1.msh", verbose=True)
bound = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}

for n in m.nodesOnLine(bound.keys()):
    n.calc = False

nodes = solver.split([n for n in m.nodes if n.calc], 1E7, 2)
colors = []
groups = []
for g, dofs in enumerate(nodes):
    groups.append([])
    c = 0
    while len(dofs) > 0:
        groups[g].append([])
        groups[g][c] = [dofs.pop()]
        elements = [e for n in groups[g][c] for e in m.elements]
        for i, n in enumerate(dofs):
            if len(set(n.elements).intersection(elements)) is 0:
                groups[g][c].append(dofs.pop(i))
                elements = [e for n in groups[g][c] for e in n.elements]
        print "Color: %i, %i nodes." % (c, len(groups[g][c]))
        c += 1
