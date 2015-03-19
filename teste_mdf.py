#!/usr/bin/python
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
__version__ = "0.1.00"
__author__ = "kewitz"
__license__ = "MIT"
DEBUG = True

from os import path
from ctypes import cdll, byref, c_double
from numpy import zeros, array, ctypeslib
import matplotlib.pyplot as plt
import time


def timeit(t=False):
    return time.time() - t if t else time.time()

_dir = path.dirname(__file__)
mdf = cdll.LoadLibrary(path.join(_dir, 'mdf.so'))

alpha = 1.73
iteracoes = 4000
w, h = 300, 300
X = zeros((h, w))
bound = array([100.00, 33.33, 66.66, 0.0])

t = timeit()
mdf.run(w, h, iteracoes, c_double(alpha), byref(ctypeslib.as_ctypes(bound)),
        byref(ctypeslib.as_ctypes(X)))
print "[!] Solve GPU: %.3f ms" % timeit(t)

t = timeit()
mdf.runCPU(w, h, iteracoes, c_double(alpha), byref(ctypeslib.as_ctypes(bound)),
           byref(ctypeslib.as_ctypes(X)))
print "[!] Solve CPU: %.3f ms" % timeit(t)

h = plt.contour(X)
plt.clabel(h, inline=1, fontsize=10)