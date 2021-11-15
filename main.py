####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Email	: mengchil@cs.cmu.edu      #
####################################

import os
import sys
import copy
import progressbar
import time
import numpy as np
import collections
from termcolor import colored

from config import parser
from gspan import *
from utils import *

from graph import AUTO_EDGE_ID
from graph import Graph, instance_filter_mis
from graph import VACANT_GRAPH_ID
from graph import VACANT_VERTEX_LABEL

def main(FLAGS=None):
	"""Run GRANDDIWE"""

	if FLAGS is None:
		FLAGS, _ = parser.parse_known_args(args=sys.argv[1:])

		graphs = read_graphs(FLAGS.input_file)
		graph_ori_mdl, a_score, g_mdl_bits, m_mdl_bits, w_mdl_bits = [], [], [], [], []
		for idx, _ in enumerate(graphs):
			graph_ori_mdl.append(graphs[idx].calculate_mdl())
			a_score.append([])
		total_bits = [sum(graph_ori_mdl)]

		iter, tttime = 0, 0
		min_s = FLAGS.min_support
		while True:
			print('\n----------------------------------------------------------------------\n')
			print(colored('Iteration ' + str(iter) + '\n', 'magenta'))
			print(colored('Finding Frequent Substructures...', 'blue'))
			start1 = time.time()
			gs = gSpan(
				graphs=copy.deepcopy(graphs),
				iteration = iter,
				min_support=min_s,
				min_num_vertices=FLAGS.lower_bound_of_num_vertices,
				max_num_vertices=FLAGS.upper_bound_of_num_vertices,
				verbose=FLAGS.verbose,
				visualize=FLAGS.plot,
				where=FLAGS.where
			)
			freq_subs = gs.run()
			freq_subs = eliminate_incorrect_pattern(copy.deepcopy(freq_subs), min_s)
			freq_subs = find_directed_frequent_subgraphs(freq_subs, min_s)
			freq_subs = instance_filter_mis(freq_subs, min_s)
			freq_subs = self_loop_detect(freq_subs, graphs, min_s)
			end1 = time.time()
			print(colored('Finishedd in %f Seconds!\n' % (end1 - start1), 'green'))

			print(colored('Optimizing Edge Weight...', 'blue'))
			start2 = time.time()
			for idx, fs in enumerate(progressbar.progressbar(freq_subs)):
				fs.update_edge_weight(graphs)
				fs.update_gdict()
			end2 = time.time()
			print(colored('Finished in %f Seconds!\n' % (end2 - start2), 'green'))

			print(colored('Evaluting Substructures...', 'blue'))
			start3 = time.time()
			max_score = 0
			max_score_idx = -1
			max_arr, max_mdl = [], dict()
			for idx, fs in enumerate(progressbar.progressbar(freq_subs)):
				score, tmp_arr, tmp_mdl = 0, [], dict()
				fs_gid = [gi.gid for gi in fs.instances]
				for gid in fs_gid:
					g = graphs[gid]
					g_new_mdl = g.calculate_compressed_mdl(fs)
					g_old_mdl = g.calculate_mdl()

					# if MDL doesn't become smaller, then don't compress it (even if it contains pattern)
					if g_old_mdl - g_new_mdl > 0:
						score += g_old_mdl - g_new_mdl
						tmp_arr.append(gid)
						tmp_mdl[gid] = (g_old_mdl, g_new_mdl)
				f_mdl = fs.definition.calculate_mdl()

				if max(0, score - f_mdl) > max_score and len(tmp_arr) > 0:
					max_score = score - f_mdl
					max_score_idx = idx
					max_arr = copy.deepcopy(tmp_arr)
					max_mdl = copy.deepcopy(tmp_mdl)

			end3 = time.time()
			print(colored('Finished in %f Seconds!\n' % (end3 - start3), 'green'))
			if max_score_idx == -1:
				min_s = int(min_s * 0.9)
				if min_s <= 30:
					ttime = (end1 - start1) + (end2 - start2) + (end3 - start3)
					tttime += ttime
					print(colored('No More Pattern is Found!', 'green'))
					print(colored(str(iter) + ' Iterations Finished in %f Seconds!\n' % (tttime), 'green'))
					break
				else:
					print(colored('Decrease MinSup', 'red'))
					continue

			freq_subs[max_score_idx].display()
			if FLAGS.plot:
				draw_graph(g=freq_subs[max_score_idx].definition,
						   path=FLAGS.where,
						   label='Pattern ' + str(iter) + '\n# of Instances: ' + str(len(fs.instances)))

			print(colored('Compressing Graphs...', 'blue'))
			start4 = time.time()
			ins_count = 0
			total_bits.append(total_bits[-1])
			for idx, _ in enumerate(progressbar.progressbar(graphs)):
				if idx in max_arr:
					g_old_mdl, g_new_mdl = max_mdl[idx]
					a_score[idx].append((g_old_mdl - g_new_mdl) / graph_ori_mdl[idx])
					ins_num, ins_count = graphs[idx].compress(freq_subs[max_score_idx], iter, ins_count)
					total_bits[iter + 1] -= g_old_mdl - g_new_mdl
				else:
					a_score[idx].append(0)
			end4 = time.time()
			print(colored('Finished in %f Seconds!\n' % (end4 - start4), 'green'))

			ttime = (end1 - start1) + (end2 - start2) + (end3 - start3) + (end4 - start4)
			tttime += ttime
			print(colored('Iteration ' + str(iter) + ' Finished in %f Seconds\n' % ttime, 'magenta'))

			iter += 1
			if iter == FLAGS.iterations:
				print(colored(str(FLAGS.iterations) + ' Iterations Finished in %f Seconds!\n' % (tttime), 'green'))
				break

		with open(FLAGS.output_file, 'w') as f:
			f.write('MinSup ' + str(min_s) + ', Bits ' + str(total_bits[-1]) + '\n')
			for gid, i in enumerate(a_score):
				tmp = 1
				for idx, j in enumerate(i):
					tmp -= (1 / iter) * (iter - idx) * j
				f.write(str(round(tmp, 5)))
				f.write('\n')

if __name__ == '__main__':
	main()
