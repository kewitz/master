# -*- coding: utf-8 -*-
from numpy import *
import NScheme as ns

m = ns.Mesh(file="""/home/leo/Documents/Master/Pesquisa/TesteJacobiCUDA/teste1.msh""",
            verbose=True)

for n in m.nodesOnLine([1, 2, 3, 4]):
    n.calc = False

V = m.run(1000, boundary={1: 100.0, 2: 0.0, 3: 0.0, 4: 0.0})
m.plotResult(result=V)
