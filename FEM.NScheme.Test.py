# -*- coding: utf-8 -*-
from numpy import *
import NScheme as ns
import matplotlib.pyplot as plt

path = """./res/"""

bench = []
for teste in ["teste1_1.msh", "teste1_2.msh", "teste1_3.msh"]:
    m = ns.Mesh(file=path + teste)
    print "[!] Loaded %s with %i nodes." % (teste, len(m.nodes))

    for n in m.nodesOnLine([1, 2, 3, 4]):
        n.calc = False

    tc = ns.timeit()
    ic, V = m.run(cuda=False, boundary={1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0})
    tc = ns.timeit(tc)
    print "[!] Solve CPU: %i iterations in %.3fms." % (ic, tc)

    tg = ns.timeit()
    ig, V2 = m.run(cuda=True, boundary={1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0})
    tg = ns.timeit(tg)
    print "[!] Solve GPU: %i iterations in %.3fms." % (ig, tg)

    bench.append((tc, tg, ic, ig, len(m.nodes)))
    del m

# %% Plot
bench = matrix(bench)
tc = bench[:,0]
tg = bench[:,1]
ic = bench[:,2]
ig = bench[:,3]
nodes = bench[:,4].T.A[0]
width = 1000

plt.figure(1)
plt.title("Function Benchmark\n(Less is better)")
plt.bar(nodes, tc, width, color="b", label="CPU")
plt.bar(nodes+(width*1.1), tg, width, color="g", label="GPU")
plt.xlabel("Number of Nodes")
plt.ylabel("Time [s]")
plt.xticks(nodes+(width), nodes)
plt.legend(fontsize="medium", fancybox=True, framealpha=.5)

plt.figure(2)
plt.title("Function Benchmark\n(Less is better)")
plt.plot(nodes, ic, label="CPU")
plt.plot(nodes, ig, label="GPU")
plt.xlabel("Number of Nodes")
plt.ylabel("Number of Iterations")
plt.xticks(nodes)
plt.legend(fontsize="medium", fancybox=True, framealpha=.5)
