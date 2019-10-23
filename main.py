"""The main program that runs gSpan."""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import codecs
import copy
import progressbar
import time
import numpy as np
import collections
from termcolor import colored
from graphviz import Digraph

from config import parser
from gspan import gSpan, find_directed_frequent_subgraphs, eliminate_incorrect_pattern

from graph import AUTO_EDGE_ID
from graph import Graph, instance_filter_mis
from graph import VACANT_GRAPH_ID
from graph import VACANT_VERTEX_LABEL

label_type = {'1': 'OE', '2': 'SOA', '3': 'Other', '4': 'SOL', '5': 'Cash', '6': 'NOGL', '7': 'LOA', '8': 'ORS', '9': 'SFL', '10': 'LOL', '11': 'LFL'}

def draw_graph(fs, iter):
	G = Digraph()
	G.attr(label='Pattern ' + str(iter) + '\n# of Instances: ' + str(len(fs.instances)), fontsize='20')
	nodes = {}
	edges = collections.defaultdict(int)
	for e in fs.definition.edges.values():
		s, d = str(e.frm), str(e.to)
		try:
			nodes[s] = label_type[fs.definition.vertices[e.frm].vlb]
		except:
			nodes[s] = str(fs.definition.vertices[e.frm].vlb)
		try:
			nodes[d] = label_type[fs.definition.vertices[e.to].vlb]
		except:
			nodes[d] = str(fs.definition.vertices[e.to].vlb)
		edges[(s, d)] = e.weight
	for idx, n in enumerate(nodes.keys()):
		G.node(n, n + ' (' + nodes[n] + ')')
	for e in edges.keys():
		G.edge(e[0], e[1], str(edges[e]))
	G.render(os.path.join('pattern', 'pattern-' + str(iter)), format='png', cleanup=True)

def read_graphs(database_file_name, max_ngraphs):
	graphs = dict()
	with codecs.open(database_file_name, 'r', 'utf-8') as f:
		lines = [line.strip() for line in f.readlines()]
		tgraph, graph_cnt = None, 0
		for i, line in enumerate(lines):
			cols = line.split(' ')
			if cols[0] == 't':
				if tgraph is not None and len(tgraph.edges) != 0:
					graphs[graph_cnt] = tgraph
					graph_cnt += 1
					tgraph = None
				if cols[-1] == '-1' or graph_cnt >= max_ngraphs:
					break
				tgraph = Graph(graph_cnt, eid_auto_increment=True)
			elif cols[0] == 'v':
				tgraph.add_vertex(int(cols[1]), cols[2])
			elif cols[0] == 'e' and cols[1] != cols[2]:
				tgraph.add_edge(AUTO_EDGE_ID, int(cols[1]), int(cols[2]), 1, int(cols[3]))
	return graphs

def main(FLAGS=None):
	"""Run gSpan."""

	if FLAGS is None:
		FLAGS, _ = parser.parse_known_args(args=sys.argv[1:])

	graphs = read_graphs(os.path.join('data', FLAGS.database_file_name, 'data.txt'), FLAGS.num_graphs)
	graph_ori_mdl = []
	a_score = []
	for idx, _ in enumerate(graphs):
		graph_ori_mdl.append(graphs[idx].calculate_mdl())
		a_score.append([])

	iter = 0
	tttime = 0
	while True:
		print('\n----------------------------------------------------------------------\n')
		print(colored('Iteration ' + str(iter) + '\n', 'magenta'))
		print(colored('Finding Frequent Substructures...', 'blue'))
		start1 = time.time()
		gs = gSpan(
			graphs=copy.deepcopy(graphs),
			iteration = iter,
			min_support=FLAGS.min_support,
			min_num_vertices=FLAGS.lower_bound_of_num_vertices,
			max_num_vertices=FLAGS.upper_bound_of_num_vertices,
			verbose=FLAGS.verbose,
			visualize=FLAGS.plot,
			where=FLAGS.where
		)
		undir_freq_subs = gs.run()
		undir_freq_subs = eliminate_incorrect_pattern(copy.deepcopy(undir_freq_subs))
		dir_freq_subs = find_directed_frequent_subgraphs(undir_freq_subs, FLAGS.min_support)
		freq_subs = instance_filter_mis(dir_freq_subs, FLAGS.min_support)
		end1 = time.time()
		print(colored('Finishedd in %f Seconds!\n' % (end1 - start1), 'green'))

		print(colored('Optimizing Edge Weight...', 'blue'))
		start2 = time.time()
		for fs in progressbar.progressbar(freq_subs):
			fs.update_edge_weight(graphs)
		end2 = time.time()
		print(colored('Finished in %f Seconds!\n' % (end2 - start2), 'green'))

		print(colored('Evaluting Substructures...', 'blue'))
		start3 = time.time()
		max_score = 0
		max_score_idx = -1
		max_arr = []
		for idx, fs in enumerate(progressbar.progressbar(freq_subs)):
			score = 0
			tmp_arr = []
			for gid in fs.gid_set:
				g = graphs[gid]
				c_mdl = g.calculate_compressed_mdl(fs)
				g_mdl = g.calculate_mdl()
				### if MDL doesn't become smaller, then don't compress it (even if it contains pattern)
				if g_mdl - c_mdl > 0:
					score += g_mdl - c_mdl
					tmp_arr.append(gid)
			f_mdl = fs.definition.calculate_mdl()
			if max(0, score - f_mdl) > max_score:
				max_score = score - f_mdl
				max_score_idx = idx
				max_arr = tmp_arr
		end3 = time.time()
		print(colored('Finished in %f Seconds!\n' % (end3 - start3), 'green'))

		if max_score_idx == -1:
			if FLAGS.min_support <= 5:
				ttime = (end1 - start1) + (end2 - start2) + (end3 - start3)
				tttime += ttime
				print(colored('No More Pattern is Found!', 'green'))
				print(colored(str(iter) + ' Iterations Finished in %f Seconds!\n' % (tttime), 'green'))
				break
			else:
				FLAGS.min_support = int(FLAGS.min_support * 0.95)
				continue

		freq_subs[max_score_idx].display()
		# draw_graph(freq_subs[max_score_idx], iter)

		print(colored('Compressing Graphs...', 'blue'))
		start4 = time.time()
		ins_count = 0
		for idx, _ in enumerate(progressbar.progressbar(graphs)):
			if idx in max_arr:
				prev = graphs[idx].calculate_mdl()
				ins_num, ins_count = graphs[idx].compress(freq_subs[max_score_idx], iter, ins_count)
				a_score[idx].append((prev - graphs[idx].calculate_mdl()) / graph_ori_mdl[idx])
			else:
				a_score[idx].append(0)
		end4 = time.time()
		print(colored('Finished in %f Seconds!\n' % (end4 - start4), 'green'))

		ttime = (end1 - start1) + (end2 - start2) + (end3 - start3) + (end4 - start4)
		tttime += ttime
		print(colored('Iteration ' + str(iter) + ' Finished in %f Seconds\n' % ttime, 'magenta'))


		iter += 1
		with open('results/' + FLAGS.database_file_name + '/ascore-' + str(iter) + '.txt', 'w') as f:
			f.write('MinSup ' + str(FLAGS.min_support) + '\n')
			for i in a_score:
				tmp = 1
				for idx, j in enumerate(i):
					tmp -= (1 / iter) * (iter - idx) * j
				f.write(str(round(tmp, 5)))
				f.write('\n')
		np.savetxt('mdl/mdl-' + str(iter) + '.txt', a_score, fmt='%f', delimiter='\n')

		if iter == FLAGS.iterations:
			print(colored(str(FLAGS.iterations) + ' Iterations Finished in %f Seconds!\n' % (tttime), 'green'))
			break

if __name__ == '__main__':
	main()
