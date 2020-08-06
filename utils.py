####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Date	: 2020/05/02               #
####################################

import codecs
import os
from graphviz import Digraph
from sklearn import metrics
from sklearn.metrics import average_precision_score
from graph import *

def draw_graph(g, path, label):
	G = Digraph()
	G.attr(label=label, fontsize='20')
	nodes = {}
	edges = collections.defaultdict(int)
	for e in g.edges.values():
		s, d = str(e.frm), str(e.to)
		try:
			nodes[s] = enron_type[g.vertices[e.frm].vlb]
		except:
			nodes[s] = str(g.vertices[e.frm].vlb)
		try:
			nodes[d] = enron_type[g.vertices[e.to].vlb]
		except:
			nodes[d] = str(g.vertices[e.to].vlb)
		edges[(s, d)] = e.weight
	for idx, n in enumerate(nodes.keys()):
		G.node(n, n + ' (' + nodes[n] + ')')
	for e in edges.keys():
		G.edge(e[0], e[1], str(edges[e]))
	G.render(path, format='png', cleanup=True)

def read_graphs(database_file_name):
	graphs = dict()
	with codecs.open(database_file_name, 'r', 'utf-8') as f:
		lines = [line.strip() for line in f.readlines()]
		tgraph, graph_cnt = None, 0
		for i, line in enumerate(lines):
			cols = line.split(' ')
			if cols[0] == 't':
				if tgraph is not None:
					graphs[graph_cnt] = tgraph
					graph_cnt += 1
					tgraph = None
				if cols[-1] == '-1':
					break
				tgraph = Graph(graph_cnt, eid_auto_increment=True)
			elif cols[0] == 'v':
				tgraph.add_vertex(int(cols[1]), cols[2])
			elif cols[0] == 'e':
				tgraph.add_edge(AUTO_EDGE_ID, int(cols[1]), int(cols[2]), 1, int(cols[3]))
	return graphs
