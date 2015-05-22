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
import time
import sys
from numpy import *
from ctypes import *
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Importa a biblioteca CUDA.
lib = cdll.LoadLibrary('./nscheme.so')
lib.hello()
#assert lib.getCUDAdevices() > 0, "No CUDA capable devices found."

eps = 8.854187E-12


def timeit(t=False):
    """Função cronômetro."""
    return time.time() - t if t else time.time()


class _node(Structure):
    """Struct `node` utilizada pelo programa C."""
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("i", c_uint),
                ("calc", c_bool),
                ("ne", c_uint),
                ("elements", c_uint*10)]


class _elementri(Structure):
    """Struct `element` utilizada pelo programa C."""
    _fields_ = [("nodes", c_uint*3),
                ("matriz", c_float*6),
                ("eps", c_float)]


class _color(Structure):
    """Struct `color` utilizada pelo programa C."""
    _fields_ = [("len", c_uint),
                ("nodes", POINTER(c_uint))]


class Node(object):
    """Classe do nó."""
    def __init__(self, *args):
        assert len(args[0]) == 4, "{0} need 4 values".format(args)
        i, x, y, z = args[0]
        self.i = int(i)-1
        self.x, self.y = float(x), float(y)
        self.p = array([self.x, self.y])
        self.elements = []
        self.calc = True

    @property
    def ctyped(self):
        """Retorna o Nó em formato `Struct _node`."""
        r = _node(self.x, self.y, self.i, self.calc, len(self.elements))
        for i, e in enumerate(self.elements):
            r.elements[i] = e
        return r


class Element(object):
    """Classe do elemento."""
    def __init__(self, *args, **kwargs):
        x = args[0]
        i, typ, ntags = x[:3]
        self.i, self.dim = int(i)-1, int(typ)
        self.eps = eps
        self.tags = [int(a) for a in x[3:3+int(ntags)]]
        # If supplied the node list make reference, else use only the index.
        if 'nodes' in kwargs:
            self.nodes = [kwargs['nodes'][int(a)-1] for a in x[3+int(ntags):]]
            if self.dim == 2:
                # If element is bi-dimensional, add the element index to
                # its nodes related elements list.
                for n in self.nodes:
                    n.elements.append(self.i)
        else:
            self.nodes = [int(a) for a in x[3+int(ntags):]]
        assert len(self.nodes) <= 3, "You can only use triangular elements."

    @property
    def ctyped(self):
        """Retorna o Elemento em formato `Struct _elementri`."""
        r = _elementri()
        r.eps = float32(self.eps)
        for i, n in enumerate(self.nodes):
            r.nodes[i] = n.i
        return r


class Mesh(object):
    """Classe da malha."""
    verbose = False

    def __init__(self, verbose=False, debug=False, **kwargs):
        """
        Read a Gmsh `file` and parse all nodes and elements.
        """
        assert 'file' in kwargs, "File not specified."
        self.debug = debug
        if self.debug:
            t = timeit()
        # Read and parse the Gmsh file.
        with open(kwargs['file']) as f:
            x = f.read()
            ns, ne = x.find('$Nodes\n'), x.find('$EndNodes\n')
            nodes = map(lambda n: n.split(), x[ns+7:ne].split('\n')[1:-1])
            es, ee = x.find('$Elements\n'), x.find('$EndElements\n')
            elements = map(lambda n: n.split(), x[es+10:ee].split('\n')[1:-1])
        # Map nodes and Elements.
        self.nodes = map(lambda x: Node(x), nodes)
        self.elements = map(lambda x: Element(x, nodes=self.nodes), elements)
        # Verbosity...
        if verbose:
            self.verbose = verbose
            print "Done parsing {0} nodes and {1} elements."\
                  .format(len(nodes), len(elements))
        if self.debug:
            print "[!] Parse took %.3f ms" % timeit(t)
            print "[!] Using %.2f KB" % (sys.getsizeof(self)/(1024))

    def __sizeof__(self):
        return sys.getsizeof(self.elements) + sys.getsizeof(self.nodes)

    def run(self, R=0, errmin=1E-5, kmax=10000, cuda=False, **kwargs):
        """Run simulation until converge to `alpha` residue."""
        if cuda:
            assert lib.getCUDAdevices() > 0, "No CUDA capable devices found."
            func = lib.runGPU
        else:
            func = lib.runCPU
        ne, nn = len(self.elements), len(self.nodes)
        V = zeros(nn, dtype=float32)
        bench = zeros(3, dtype=float32)

        # Get ctyped elements.
        c_elements = _elementri * ne
        elements = c_elements()
        for i, e in enumerate(self.elements):
            elements[i] = e.ctyped
        # Get ctyped nodes.
        c_nodes = _node * nn
        nodes = c_nodes()
        for i, n in enumerate(self.nodes):
            nodes[i] = n.ctyped
        # Set up the boundary information.
        if 'boundary' in kwargs:
            for k in kwargs['boundary']:
                for i in self.nodesOnLine(k, True):
                    V[i] = kwargs['boundary'][k]
        # Set up colors.
        colors = m.coloring()
        c_color = ns._color * len(colors)
        ccolors = c_color()
        ccs = []
        for i, co in enumerate(colors):
            cs = array(co, dtype=uint32)
            ccs.append(cs)
            ccolors[i] = ns._color(c_uint(len(co)), ctypeslib.as_ctypes(ccs[i]))

        iters = func(ne, nn, kmax, c_float(R), c_float(errmin), elements,
                     nodes, byref(ctypeslib.as_ctypes(V)), self.verbose,
                     byref(ctypeslib.as_ctypes(bench)))

        return V, iters, bench.tolist()

    def nodesOnLine(self, tags, indexOnly=False):
        """
        Return the list of nodes that are in elements tagged with `tags`.
        Used to get boundary elements.
        """
        tags = [tags] if type(tags) is int else tags
        r = [ns for tag in tags for el in self.elements
             for ns in el.nodes if el.dim == 1
             and tag in el.tags]
        if indexOnly:
            r = [n.i for n in r]
        return list(set(r))

    def coloring(self):
        colors = []
        mapped = []

        nodes = filter(lambda n: n.calc is False, self.nodes)

        while len(nodes) != 0:
            nodes = list(set(nodes).difference(mapped))
            mapped = mapped + nodes

            ids = [n.i for n in nodes]
            if len(ids) != 0:
                colors.append(ids)

            elements = []
            for n in [n.elements for n in nodes]:
                elements = elements + n
            elements = [self.elements[i] for i in list(set(elements))]

            nodes = []
            for e in elements:
                nodes = nodes + e.nodes

        assert sum([len(c) for c in colors]) == len(self.nodes)
        return colors

    def elementsByTag(self, tags):
        """Return elements tagged by `tag`."""
        tags = [tags] if type(tags) is int else tags
        e = [el for tag in tags for el in self.elements if el.dim == 2
             and tag in el.tags]
        return list(set(e))

    def triangulate(self):
        points = array([(n.x, n.y) for n in self.nodes])
        x = points[:, 0]
        y = points[:, 1]
        return tri.Triangulation(x, y)

    def plotMesh(self, **kwargs):
        fig, ax = plt.subplots()
        if 'result' in kwargs:
            self.plotResult(result=kwargs['result'])
        elements = kwargs['elements'] if 'elements' in kwargs\
            else filter(lambda el: el.dim == 2, self.elements)
        for e in elements:
            c = e.color if 'color' in e.__dict__ else 'k'
            p = Polygon(map(lambda no: (no.x, no.y), e.nodes), closed=True,
                        fill=False, ec=c, linewidth=.5, alpha=.6)
            ax.add_patch(p)
        ax.axis('equal')
        plt.show()

    def plotResult(self, **kwargs):
        t = self.triangulate()
        plt.tricontourf(t, kwargs['result'], 15, cmap=plt.cm.rainbow)
        plt.colorbar()
        plt.show()
