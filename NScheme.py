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
lib.alloc._restype_ = c_uint
#assert lib.getCUDAdevices() > 0, "No CUDA capable devices found."


def timeit(t=False):
    """Função cronômetro."""
    return time.clock() - t if t else time.clock()


def split(array, limit, minStacks=False):
    r = []
    if minStacks:
        limit = 1+len(array)/minStacks
    stacks = 1+len(array)/limit
    for i in range(stacks):
        r.append(array[limit*i:limit*(i+1)])
    return r


class _element(Structure):
    """Struct `element` utilizada pelo programa C."""
    _fields_ = [("nodes", c_uint*3),
                ("matriz", c_float*6),
                ("mat", c_float),
                ("x", c_float*3),
                ("y", c_float*3)]


class _node(Structure):
    """Struct `node` utilizada pelo programa C."""
    _fields_ = [("i", c_uint),
                ("ne", c_uint),
                ("elements", c_uint*10)]


class _group(Structure):
    _fields_ = [("nn", c_uint),
                ("ne", c_uint),
                ("nc", c_uint),
                ("nodes", POINTER(_node)),
                ("elements", POINTER(_element)),
                ("colors", POINTER(c_uint)),
                ("cnodes", POINTER(c_uint))]


class Group(object):
    def __init__(self, colors, mesh):
        self.nodes = [node for color in g for node in color]
        self.elements = mesh.getElements(self.nodes)
        self.colors = colors
        self.mesh = mesh

    @property
    def ctyped(self):
        mesh = self.mesh
        nn, ne, nc = len(self.nodes), len(self.elements), len(self.colors)

        elements = (_element * ne)()
        for i, e in enumerate(self.elements):
            elements[i] = e.ctyped
        nodes = (_node * nn)()
        for i, n in enumerate(self.nodes):
            _n = n.ctyped
            for j, e in enumerate(n.elements):
                _n.elements[j] = self.elements.index(mesh.elements[e])
            nodes[i] = _n
        colors = (_color * nc)()
        color_indexes = []
        for i, c in enumerate(self.colors):
            _c = _color(len(c))
            indxs = (c_uint*len(c))()
            for j, n in enumerate(c):
                indxs[j] = self.nodes.index(n)
            color_indexes.append(indxs)
            _c.nodes = color_indexes[i]
            colors[i] = _c
        return _group(nn, ne, nc, nodes, elements, colors)


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
        r = _node(self.i, len(self.elements))
        for i, e in enumerate(self.elements):
            r.elements[i] = e
        return r


class Element(object):
    """Classe do elemento."""
    def __init__(self, *args, **kwargs):
        x = args[0]
        i, typ, ntags = x[:3]
        self.i, self.dim = int(i)-1, int(typ)
        self.mat = 1.0
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
        assert max([len(n.elements) for n in self.nodes if n.calc]) <= 10,\
            "Node belongs to more than 10 elements."

    @property
    def ctyped(self):
        """Retorna o Elemento em formato `Struct _element`."""
        r = _element()
        r.mat = c_float(self.mat)
        for i, n in enumerate(self.nodes):
            r.nodes[i] = n.i
            r.x[i] = n.x
            r.y[i] = n.y
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
            print "[!] Done parsing {0} nodes and {1} elements."\
                  .format(len(nodes), len(elements))
        if self.debug:
            print "[!] Parse took %.3f ms" % timeit(t)
            print "[!] Using %.2f KB" % (sys.getsizeof(self)/(1024))

    def __sizeof__(self):
        return sys.getsizeof(self.elements) + sys.getsizeof(self.nodes)

    @property
    def DOF(self):
        return len([n for n in self.nodes if n.calc])

    def run(self, R=0, errmin=1E-5, kmax=10000, cuda=False, coloring=False,
            mingroups=1, **kwargs):
        """Run simulation until converge to `alpha` residue."""
        # Assertions and function setting.
        if cuda is True:
            assert lib.getCUDAdevices() > 0, "No CUDA capable devices found."
            func = lib.runGPU
        else:
            func = lib.runCPU
        # Set up constants and other variables.
        NN = len(self.nodes)
        V = zeros(NN, dtype=float32)
        S = zeros(NN, dtype=float32)
        bench = zeros(3, dtype=float32)

        # Set up the boundary information.
        if 'boundary' in kwargs:
            for k in kwargs['boundary']:
                for n in self.nodesOnLine(k):
                    V[n.i] = kwargs['boundary'][k]
                    n.calc = False

        limit = lib.alloc(NN)
        if coloring:
            ngs = self.getColors(limit, mingroups)
        else:
            ngs = [split([n for n in self.nodes if n.calc], limit, mingroups)]
        ng = len(ngs)
        nc = max([len(c) for c in ngs])

        groups = (_group*ng)()
        for i, group in enumerate(ngs):
            nodes = [node for color in group for node in color]
            elements = self.getElements(nodes)
            # Populate Elements
            groups[i] = _group(len(nodes), len(elements), len(group))
            groups[i].elements = (_element * len(elements))()
            for j, e in enumerate(elements):
                groups[i].elements[j] = e.ctyped
            # Populate Nodes
            groups[i].nodes = (_node * len(nodes))()
            for j, n in enumerate(nodes):
                _n = n.ctyped
                for k, e in enumerate(n.elements):
                    _n.elements[k] = elements.index(self.elements[e])
                groups[i].nodes[j] = _n
            # Populate Colors
            l = 0
            groups[i].colors = (c_uint * len(group))()
            groups[i].cnodes = (c_uint * len(nodes))()
            for j, color in enumerate(group):
                groups[i].colors[j] = len(color)
                for k, n in enumerate(color):
                    groups[i].cnodes[l] = nodes.index(n)
                    l += 1
        if self.verbose:
            print "[!] Started processing."
        t = timeit()
        # Call function.
        iters = func(len(ngs), NN, nc, kmax, c_float(R),
                     c_float(errmin), groups, byref(ctypeslib.as_ctypes(V)),
                     self.verbose, byref(ctypeslib.as_ctypes(bench)))
        bg = timeit(t)
        return V, iters, bg

    def getColors(self, limit, mingroups=1):
        if self.debug:
            t = timeit()
        if self.verbose:
            print "[!] Creating colors..."
        nodes = split([n for n in self.nodes if n.calc], limit, mingroups)
        groups = []
        for g, dofs in enumerate(nodes):            
            groups.append([])
            g_elements = [set()]
            for i, n in enumerate(dofs):
                elements = set(n.elements)
                new = True
                for c, color in enumerate(groups[g]):
                    if len(elements.intersection(g_elements[c])) is 0:
                        groups[g][c].append(n)
                        g_elements[c] = g_elements[c].union(elements)
                        new = False
                        break
                if new:
                    groups[g].append([n])
                    g_elements.append(elements)

            if self.debug:
                print "[!] Group %i with %i colors." %(g, len(groups[g]))
        if self.debug:
            print "[!] Done in %.3f seconds." % timeit(t)
        return groups

    def makeColors(self, limit, bounds, mingroups=1):
        for k in bounds:
            for n in self.nodesOnLine(k):
                n.calc = False
        return self.getColors(limit, mingroups)
        
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

    def getElements(self, nodes):
        return [e for e in self.elements
                if len(set(e.nodes).intersection(nodes)) > 0
                and e.dim is 2]

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

    def plotMesh(self, result=False, **kwargs):
        fig, ax = plt.subplots()
        if 'result' is not False:
            self.plotResult(result)
        elements = kwargs['elements'] if 'elements' in kwargs\
            else filter(lambda el: el.dim == 2, self.elements)
        for e in elements:
            c = e.color if 'color' in e.__dict__ else 'k'
            p = Polygon(map(lambda no: (no.x, no.y), e.nodes), closed=True,
                        fill=False, ec=c, linewidth=.5, alpha=.6)
            ax.add_patch(p)
        ax.axis('equal')
        plt.show()

    def plotResult(self, result):
        t = self.triangulate()
        plt.tricontourf(t, result, 15, cmap=plt.cm.rainbow)
        plt.colorbar()
        plt.show()
