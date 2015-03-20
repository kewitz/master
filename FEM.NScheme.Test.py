# -*- coding: utf-8 -*-
from numpy import *
import NScheme as ns

path = """/home/leo/Documents/Master/Pesquisa/TesteJacobiCUDA/"""

for teste in ["teste1_1.msh", "teste1_2.msh"]:
    m = ns.Mesh(file=path + teste)

    for n in m.nodesOnLine([1, 2, 3, 4]):
        n.calc = False

    t = ns.timeit()
    i, V = m.run(cuda=False, boundary={1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0})
    print "[!] Solve CPU: %i iterations in %.3fms." % (i, ns.timeit(t))

    t = ns.timeit()
    i, V2 = m.run(cuda=True, boundary={1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0})
    print "[!] Solve GPU: %i iterations in %.3fms." % (i, ns.timeit(t))

    del m
