# -*- coding: utf-8 -*-
from numpy import *
import NScheme as ns

m = ns.Mesh(file="""/home/leo/Documents/Master/Pesquisa/TesteJacobiCUDA/teste1.msh""",
            verbose=True)

for n in m.nodesOnLine([1, 2, 3, 4]):
    n.calc = False

V = m.run(5000, boundary={1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0})
m.plotMesh(result=V)
