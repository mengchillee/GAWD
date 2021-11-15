####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Email	: mengchil@cs.cmu.edu      #
####################################

import pickle
from graphviz import Digraph
from graph import *

def draw_graph(g, path, label):
    G = Digraph()
    G.attr(label=label, fontsize='20')
    nodes = {}
    edges = collections.defaultdict(int)
    for e in g.edges.values():
        s, d = str(e.frm), str(e.to)
        nodes[s] = str(g.vertices[e.frm].vlb)
        nodes[d] = str(g.vertices[e.to].vlb)
        edges[(s, d)] = e.weight
    for idx, n in enumerate(nodes.keys()):
        G.node(n, n + ' (' + nodes[n] + ')')
    for e in edges.keys():
        G.edge(e[0], e[1], str(edges[e]))
    G.render(path, format='png', cleanup=True)

def read_graphs(file_name):
    with open(file_name, 'rb') as handle:
        data = np.array([g[0] for g in pickle.load(handle)], dtype=object)

    graphs = dict()
    for idx, g in enumerate(data):
        tgraph = Graph(idx, eid_auto_increment=True)
        for n in g.nodes():
            tgraph.add_vertex(int(n), str(g.nodes[n]['type']))
        for e in g.edges():
            tgraph.add_edge(AUTO_EDGE_ID, int(e[0]), int(e[1]), 1, int(g.edges[e]['weight']))
        graphs[idx] = tgraph

    return graphs
