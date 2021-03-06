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
import EScheme as es

errmin = 1E-4
kmax = 10000

def test(solver):
    print "NODES\tELEM\tDOF\tPROC\tITERS\tTIME\tSPEEDUP\tERROR"
    relat = "%i\t%i\t%i\t%s\t%i\t%.2f\t%.2f\t%.2E"
    for f in files:
        m = solver.Mesh(file=f)
        nn, ne = len(m.nodes), len(m.elements)
        vc, ic, bc = m.run(cuda=False, coloring=c, boundary=bound, errmin=errmin,
                           kmax=kmax, R=R)
        print relat % (nn, ne, m.DOF, "CPU", ic, bc, 0.0, 0.0)
        m = solver.Mesh(file=f)
        vg, ig, bg = m.run(cuda=True, coloring=c, boundary=bound, errmin=errmin,
                           kmax=kmax, R=R)
        emax = max([a for a in abs((vg-vc)/vc) if str(a) != 'nan'])
        print relat % (nn, ne, m.DOF, "GPU", ig, bg, bc/bg, emax)

#%% NScheme
bound = {2: 100.0, 5: 0.0}
files = ["./res/L1.msh", "./res/L2.msh", "./res/L5.msh"]
for f in files:
    m = ns.Mesh(file=f)
    print "%s & %i & %i" % (f, len(m.nodes), len(m.elements))
    
# SOR
c = True
for R in [.6, .7, .8]:
    print "SOR R=%f" % R
    test(ns)

#%% EScheme CG
c = False
print "EScheme CG"
test(es)
