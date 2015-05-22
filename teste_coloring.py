# -*- coding: utf-8 -*-
from numpy import *
from ctypes import *
import NScheme as ns

path = """./res/"""

m = ns.Mesh(file=path + "microstrip2.msh", verbose=True)

for n in m.nodesOnLine([1, 2, 9, 10, 5]):
    n.calc = False

colors = m.coloring()
c_color = ns._color * len(colors)
ccolors = c_color()
ccs = []
for i, co in enumerate(colors):
    cs = array(co, dtype=uint32)
    ccs.append(cs)
    ccolors[i] = ns._color(c_uint(len(co)), ctypeslib.as_ctypes(ccs[i]))

ns.lib.test_colors(len(colors), ccolors)
