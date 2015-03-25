# -*- coding: utf-8 -*-
from numpy import *
import NScheme as ns
import matplotlib.pyplot as plt

path = """./res/"""

bench = {
    'nodes': [],
    'cpu': {
        'iter': [],
        'integration': [],
        'solve': []
        },
    'gpu': {
        'iter': [],
        'malloc': [],
        'integration': [],
        'solve': []
        },
    }

for teste in ["teste1_1.msh", "teste1_2.msh", "teste1_3.msh"]:
    m = ns.Mesh(file=path + teste)
    bound = {1: 100.0, 2: 66.0, 3: 33.0, 4: 0.0}
    print "[!] Loaded %s with %i nodes." % (teste, len(m.nodes))
    bench['nodes'].append(len(m.nodes))

    for n in m.nodesOnLine([1, 2, 3, 4]):
        n.calc = False

    vc, ic, bc = m.run(cuda=False, boundary=bound)
    bench['cpu']['iter'].append(ic)
    bench['cpu']['integration'].append(bc[0])
    bench['cpu']['solve'].append(bc[1])

    vg, ig, bg = m.run(cuda=True, boundary=bound)
    bench['gpu']['iter'].append(ig)
    bench['gpu']['malloc'].append(bg[0])
    bench['gpu']['integration'].append(bg[1])
    bench['gpu']['solve'].append(bg[2])

    del m

# %% Plot
width = 1000
x = array(bench['nodes'])

plt.figure(1)
plt.title("Solving Benchmark\n(Less is better)")
plt.bar(bench['nodes'], bench['cpu']['solve'], width, color="b", label="CPU")
plt.bar(x+(width*1.1), bench['gpu']['solve'], width, color="g", label="GPU")
plt.xlabel("Number of Nodes")
plt.ylabel("Time [s]")
plt.xticks(x+(width), x)
plt.yticks(bench['cpu']['solve']+bench['gpu']['solve'])
plt.legend(fontsize="medium", fancybox=True, framealpha=.5, loc=9)
plt.grid(axis="y")

y = array(bench['cpu']['integration']+bench['gpu']['integration'])
plt.figure(2)
plt.title("Element Integration Benchmark\n(Less is better)")
plt.bar(bench['nodes'], bench['cpu']['integration'], width, color="b", label="CPU")
plt.bar(x+(width*1.1), bench['gpu']['integration'], width, color="g", label="GPU")
plt.xlabel("Number of Nodes")
plt.ylabel("Time [ms]")
plt.xticks(x+(width), x)
plt.yticks(y, (y*1000).round(3))
plt.legend(fontsize="medium", fancybox=True, framealpha=.5, loc=9)
plt.grid(axis="y")