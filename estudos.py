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


def test(solver):
    print "DOF\tPROC\tITERS\tTIME\tERROR"
    relat = "%i\t%s\t%i\t%.2f\t%.2E"
    for f in files:
        m = solver.Mesh(file=f)
        vc, ic, bc = m.run(cuda=False, coloring=c, boundary=bound, errmin=1E-4,
                           kmax=10000, R=R)
        print relat % (m.DOF, "CPU", ic, bc, 0.0)
        m = solver.Mesh(file=f)
        vg, ig, bg = m.run(cuda=True, coloring=c, boundary=bound, errmin=1E-4,
                           kmax=10000, R=R)
        emax = max([a for a in abs((vg-vc)/vc) if str(a) != 'nan'])
        print relat % (m.DOF, "GPU", ig, bg, emax)

#%% NScheme
bound = {1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0}
files = ["./res/teste1_1.msh", "./res/teste1_2.msh", "./res/teste1_3.msh",
         "./res/teste1_4.msh", "./res/teste1_5.msh"]

# Jacobi
c = False
R = 0
print "Jacobi"
test(ns)

# SOR
c = True
for R in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
    print "SOR R=%f" % R
    test(ns)

# EScheme CG
print "EScheme CG"
test(es)
