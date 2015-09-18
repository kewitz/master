# -*- coding: utf-8 -*-
"""
The MIT License (MIT)

Copyright (c) 2014 Leonardo Kewitz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import NScheme as ns

files = ["./res/L10.msh"]
#files = ["./res/L2.msh"]

def validateColors(colors):
    v = True
    for color in colors:
        for nodes in color:
            elements = [e for n in nodes for e in n.elements]
            cv = len(elements) == len(set(elements))
            v = v and cv
            
            if not cv:
                break
    assert v, "Existem nós com elementos em comum em uma mesma cor."
    

bound = {2: 100.0, 5: 0.0}
for f in files:
    m = ns.Mesh(file=f, verbose=True, debug=True)
    limit = ns.lib.alloc(len(m.nodes))
    colors = m.makeColors(limit, bound, 1)
    validateColors(colors)
    dof = len([n for n in m.nodes if n.calc])
    nodes_mapped = sum([len(c) for g in colors for c in g])
    assert dof == nodes_mapped, "Faltando nós. {} < {}".format(nodes_mapped, dof)

print "Done."