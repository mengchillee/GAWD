####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Email	: mengchil@cs.cmu.edu      #
####################################

import argparse

def str2bool(s):
	"""Convert str to bool."""
	return s.lower() not in ['false', 'f', '0', 'none', 'no', 'n']

parser = argparse.ArgumentParser()
parser.add_argument(
	'-i', '--iterations',
	type=int,
	default=1000,
	help='number of iterations, default 10'
)
parser.add_argument(
	'-s', '--min_support',
	type=int,
	default=200,
	help='min support, default 200'
)
parser.add_argument(
	'-l', '--lower_bound_of_num_vertices',
	type=int,
	default=2,
	help='int, lower bound of number of vertices of output subgraph, default 2'
)
parser.add_argument(
	'-u', '--upper_bound_of_num_vertices',
	type=int,
	default=float('inf'),
	help='int, upper bound of number of vertices of output subgraph, '
		 'default inf'
)
parser.add_argument(
	'-v', '--verbose',
	type=str2bool,
	default=False,
	help='bool, verbose output, default off'
)
parser.add_argument(
	'-f', '--input_file',
	type=str,
	help='str, graph database file name'
)
parser.add_argument(
	'-o', '--output_file',
	type=str,
	default='result.txt',
	help='str, anomaly score file name'
)
parser.add_argument(
	'-p', '--plot',
	type=str2bool,
	default=False,
	help='bool, whether plot frequent subgraph, default off'
)
parser.add_argument(
	'-w', '--where',
	type=str2bool,
	default=False,
	help='bool, output where one frequent subgraph appears in database, '
		 'default off'
)
