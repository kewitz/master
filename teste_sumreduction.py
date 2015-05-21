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

from ctypes import cdll, byref, c_double, c_int, c_float
from numpy import zeros, ones, matrix, array, ctypeslib, random, linalg, float32, arange
import time


def timeit(t=False):
    return time.time() - t if t else time.time()

lib = cdll.LoadLibrary('./escheme.so')

n = 3152
A = array(random.random(n), dtype=float32)
B = array(random.random(n), dtype=float32)

lib.teste_sum_reduction.restype = c_float
s = lib.teste_sum_reduction(n, byref(ctypeslib.as_ctypes(A)),
                            byref(ctypeslib.as_ctypes(B)))
t = sum(A*B)
print "Kernel produced: {:f} \nCorrect answer: {:f} \nDiff: {:f}".format(s, t, s-t)
