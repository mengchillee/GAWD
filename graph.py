####################################
# Author: Jeremy (Meng-Chieh) Lee  #
# Email	: mengchil@cs.cmu.edu      #
####################################

import collections
import itertools
import copy
from math import log2, ceil
from scipy.special import comb
import networkx as nx
from networkx.algorithms.approximation import independent_set
import numpy as np
import progressbar
import time

VACANT_EDGE_ID = -1
VACANT_VERTEX_ID = -1
VACANT_EDGE_LABEL = -1
VACANT_VERTEX_LABEL = -1
VACANT_GRAPH_ID = -1
AUTO_EDGE_ID = -1

def logstar(x):
	try:
		return 2 * log2(x) + 1
	except:
		return 0

def find_instance_info(ins_arr, edges):
	ins_num = 0
	ins_vertices = []
	ins_edges = []
	for idx, ins in enumerate(ins_arr):
		ins_vertices.append(set())
		ins_edges.append([])
		while ins != None:
			ins_vertices[idx].add(ins.edge.frm)
			ins_vertices[idx].add(ins.edge.to)
			if ins.edge.directed == '0':
				for i in edges.keys():
					if edges[i].to == ins.edge.frm and edges[i].frm == ins.edge.to:
						ins_edges[idx].append(i)
						break
			else:
				for i in edges.keys():
					if edges[i].frm == ins.edge.frm and edges[i].to == ins.edge.to:
						ins_edges[idx].append(i)
						break
			ins = ins.prev
			ins_num += 1
		ins_vertices[idx] = list(ins_vertices[idx])
	return np.array(ins_vertices), np.array(ins_edges)

def instance_filter_mis(dir_freq_subs, min_support):
	del_dfs_idx = []
	for dfs_idx, dfs in enumerate(progressbar.progressbar(dir_freq_subs)):
		counter = collections.Counter([ins.gid for ins in dfs.instances])
		overlap_ins_gid = [k for k, v in counter.items() if v >= 2]
		for gid in overlap_ins_gid:
			ins_idx = [idx for idx, ins in enumerate(dfs.instances) if ins.gid == gid]
			ins_node_dict = {}
			NG = nx.Graph()
			for ins, iid in zip(dfs.instances[ins_idx], ins_idx):
				ins_node = set()
				while ins:
					ins_node.add(ins.edge.frm)
					ins_node.add(ins.edge.to)
					ins = ins.prev
				ins_node_dict[iid] = set(sorted(ins_node))
				NG.add_node(iid)
			for in_key1, in_val1 in ins_node_dict.items():
				for in_key2, in_val2 in ins_node_dict.items():
					if in_key1 != in_key2 and len(in_val1.intersection(in_val2)) > 0:
						NG.add_edge(in_key1, in_key2)
			mis = independent_set.maximum_independent_set(NG)
			del_idx = list(set(ins_idx).difference(mis))
			dir_freq_subs[dfs_idx].instances = np.delete(dir_freq_subs[dfs_idx].instances, del_idx, 0)
		if len(dir_freq_subs[dfs_idx].instances) < min_support:
			del_dfs_idx.append(dfs_idx)
	dir_freq_subs = np.delete(dir_freq_subs, del_dfs_idx, 0)
	return dir_freq_subs

class Edge(object):
	def __init__(self,
				 eid=VACANT_EDGE_ID,
				 frm=VACANT_VERTEX_ID,
				 to=VACANT_VERTEX_ID,
				 elb=VACANT_EDGE_LABEL,
				 weight=VACANT_EDGE_LABEL,
				 directed='0'):
		self.eid = eid
		self.frm = frm
		self.to = to
		self.elb = elb
		self.weight = weight
		self.directed = directed

	def display(self):
		if self.directed:
			print('d {} {} {}'.format(self.frm, self.to, self.weight))
		else:
			print('u {} {} {}'.format(self.frm, self.to, self.weight))

class Vertex(object):
	def __init__(self,
				 vid=VACANT_VERTEX_ID,
				 vlb=VACANT_VERTEX_LABEL):
		self.vid = vid
		self.vlb = vlb
		self.edges = dict()
		self.directed_list = []

	def add_edge(self, eid, frm, to, elb, weight, directed):
		self.edges[to] = Edge(eid, frm, to, elb, weight, directed)
		if directed == '1':
			self.directed_list.append(to)

	def remove_edge(self, to):
		del self.edges[to]
		self.directed_list.remove(to)

	def display(self):
		print('v {} {} {}'.format(self.vid, self.vlb, self.directed_list))

class Graph(object):
	def __init__(self,
				 gid=VACANT_GRAPH_ID,
				 eid_auto_increment=True):
		self.gid = gid
		self.vertices = dict()
		self.edges = dict()
		self.eid_auto_increment = eid_auto_increment
		self.counter = itertools.count()

	def find_edge_id(self, frm, to):
		for i in self.edges.keys():
			if self.edges[i].frm == frm and self.edges[i].to == to:
				return i
		return -1

	def find_edge_directed(self, frm, to):
		eid = self.find_edge_id(frm, to)
		if eid == -1:
			return False
		else:
			return self.edges[eid].directed

	def remove_edge(self, eid):
		self.vertices[self.edges[eid].frm].remove_edge(self.edges[eid].to)
		del self.edges[eid]

	def calculate_instances_attr_num(self, subset):
		va = set()
		subset = subset.reshape(-1)
		for v in self.vertices.values():
			if v.vid not in subset:
				va.add(v.vlb)
		return len(va) + 1

	def calculate_attr_num(self):
		va = set()
		for v in self.vertices.values():
			va.add(v.vlb)
		return len(va)

	def calculate_max_edge(self, ie=None):
		max_edge_num = 0
		for e in self.edges.values():
			if ie == None or ((e.frm, e.to) not in ie):
				if e.weight > max_edge_num:
					max_edge_num = e.weight
		return max_edge_num

	def calculate_multiplicities(self, ins_edges_arr, edge_weight):
		mbits = 0
		for ins_edges in ins_edges_arr:
			for ie, iew in zip(ins_edges, edge_weight):
				diff = self.edges[ie].weight - iew
				xxx = 2 * log2(abs(diff)) + 3 if diff != 0 else 1
				mbits += xxx
		return mbits

	def generate_new_edges(self, ins_vertices, ins_edges):
		new_edges = collections.defaultdict(list)
		ver_cnt = collections.defaultdict(set)
		ie_flat = np.array(ins_edges).reshape(-1)
		for e in self.edges.values():
			sn_frm = [idx for idx, iv in enumerate(ins_vertices) if e.frm in iv]
			sn_to = [idx for idx, iv in enumerate(ins_vertices) if e.to in iv]
			if len(sn_frm) != 0 and len(sn_to) == 0:
				sn_frm_id = 's' + str(sn_frm[0])
				new_edges[(sn_frm_id, e.to)].append(e.weight)
				ver_cnt[sn_frm_id].add(e.to)
			elif len(sn_frm) == 0 and len(sn_to) != 0:
				sn_to_id = 's' + str(sn_to[0])
				new_edges[(e.frm, sn_to_id)].append(e.weight)
				ver_cnt[e.frm].add(sn_to_id)
			elif len(sn_frm) != 0 and len(sn_to) != 0 and e.eid not in ie_flat:
				sn_frm_id, sn_to_id = 's' + str(sn_frm[0]), 's' + str(sn_to[0])
				new_edges[(sn_frm_id, sn_to_id)].append(e.weight)
				ver_cnt[sn_frm_id].add(sn_to_id)
			elif len(sn_frm) == 0 and len(sn_to) == 0:
				new_edges[(e.frm, e.to)].append(e.weight)
				ver_cnt[e.frm].add(e.to)
		return new_edges, ver_cnt

	def calculate_rewiring(self, new_edges, ins_vnum):
		wbits = 0
		for (frm, to), w in new_edges.items():
			weight = sum(new_edges[(frm, to)])
			se = len(new_edges[(frm, to)])
			if 's' in str(frm) and 's' in str(to):
				wbits += 2 * log2(ins_vnum) * (se + 1) + log2(weight) * (se - 1)
			elif 's' in str(frm) or 's' in str(to):
				wbits += log2(ins_vnum) * (se + 1) + log2(weight) * (se - 1)
		return wbits

	def calculate_compressed_mdl(self, fs):
		ins = fs.find_instances(self.gid)
		ins_vertices, ins_edges = find_instance_info(ins, self.edges)

		### Multiplicities
		mbits = self.calculate_multiplicities(ins_edges, fs.edge_weight)

		v_num = len(self.vertices) - len(ins_vertices.reshape(-1)) + len(ins)
		vbits = logstar(v_num) + v_num * log2(self.calculate_instances_attr_num(ins_vertices))

		### Rewiring
		new_edges, ver_cnt = self.generate_new_edges(ins_vertices, ins_edges)
		wbits = self.calculate_rewiring(new_edges, len(ins_vertices[0]))

		### Check whether the graph is totally compressed
		if len(new_edges) != 0:
			b, s = 0, 0
			for to in ver_cnt.values():
				s += log2(comb(v_num, len(set(to))))
				if len(set(to)) > b:
					b = len(set(to))
			rbits = logstar(b) + v_num * log2(b + 1) + s

			m = max([sum(v) for v in new_edges.values()])
			ebits = logstar(m) + len(new_edges) * log2(m)
		else:
			rbits, ebits = 0, 0

		return ceil(vbits) + ceil(rbits) + ceil(ebits) + ceil(mbits) + ceil(wbits)

	def calculate_mdl(self):
		v_num = len(self.vertices)
		vbits = logstar(v_num) + v_num * log2(self.calculate_attr_num())

		b, s = 0, 0
		for v in self.vertices.values():
			sn = len(v.directed_list)
			s += log2(comb(v_num, sn))
			if sn > b:
				b = sn
		rbits = logstar(b) + v_num * log2(b + 1) + s

		ebits = 0
		if len(self.edges.values()) != 0:
			m = max([e.weight for e in self.edges.values()])
			ebits = logstar(m) + len(self.edges) * log2(m)

		return ceil(vbits) + ceil(rbits) + ceil(ebits)

	def compress(self, best_sub, iteration, ins_count):
		ins_num = 0
		def_edge_num = len(best_sub.definition.edges)
		def_vertex_num = len(best_sub.definition.vertices)

		ins_arr = best_sub.find_instances(self.gid)
		for ins in ins_arr:
			tmp = [copy.copy(ins)]
			ins_vertex, ins_edge = find_instance_info(tmp, self.edges)
			ins_vertex, ins_edge = ins_vertex[0], ins_edge[0]

			### Prevent from compressing again on the same vertex
			if len(ins_edge) != def_edge_num or len(ins_vertex) != def_vertex_num:
				continue

			ins_num += 1
			ins_count += 1
			new_vertex_label = 'PATTERN-' + str(iteration)
			new_vertex_id = max(self.vertices.keys()) + 1
			self.add_vertex(new_vertex_id, new_vertex_label)

			### Redirect
			dup_edge = {}
			del_eid_list = []
			for i in self.edges.keys():
				if i not in ins_edge:
					if self.edges[i].frm in ins_vertex and self.edges[i].to in ins_vertex:
						if (new_vertex_id, new_vertex_id) in dup_edge.keys():
							del_eid = dup_edge[(new_vertex_id, new_vertex_id)]
							self.edges[del_eid].weight += self.edges[i].weight
							del_eid_list.append(i)
						else:
							self.edges[i].frm = new_vertex_id
							self.edges[i].to = new_vertex_id
							dup_edge[(new_vertex_id, new_vertex_id)] = i
					elif self.edges[i].to in ins_vertex:
						if (self.edges[i].frm, new_vertex_id) in dup_edge.keys():
							del_eid = dup_edge[(self.edges[i].frm, new_vertex_id)]
							self.edges[del_eid].weight += self.edges[i].weight
							del_eid_list.append(i)
						else:
							self.edges[i].to = new_vertex_id
							dup_edge[(self.edges[i].frm, new_vertex_id)] = i
					elif self.edges[i].frm in ins_vertex:
						if (new_vertex_id, self.edges[i].to) in dup_edge.keys():
							del_eid = dup_edge[(new_vertex_id, self.edges[i].to)]
							self.edges[del_eid].weight += self.edges[i].weight
							del_eid_list.append(i)
						else:
							self.edges[i].frm = new_vertex_id
							dup_edge[(new_vertex_id, self.edges[i].to)] = i

			for eid in del_eid_list:
				self.remove_edge(eid)

			for vk, vv in self.vertices.items():
				del_tmp = []
				for ek, ev in vv.edges.items():
					if ev.to in ins_vertex:
						del_tmp.append(ev.to)
				for j in del_tmp:
					vv.add_edge(vv.edges[j].eid, vk, new_vertex_id, vv.edges[j].elb, vv.edges[j].weight, vv.edges[j].directed)
					vv.remove_edge(j)

			for e in self.edges.values():
				if e.frm == new_vertex_id:
					self.vertices[new_vertex_id].add_edge(e.eid, new_vertex_id, e.to, e.elb, e.weight, e.directed)

			for i in ins_edge:
				del self.edges[i]
			for i in ins_vertex:
				del self.vertices[i]

		return ins_num, ins_count

	def get_num_vertices(self):
		"""Return number of vertices in the graph."""
		return len(self.vertices)

	def add_vertex(self, vid, vlb):
		"""Add a vertex to the graph."""
		if vid in self.vertices.keys():
			return self
		self.vertices[vid] = Vertex(vid, vlb)
		return self

	def add_edge(self, eid, frm, to, elb, weight):
		"""Add an edge to the graph."""
		if (frm is self.vertices and
				to in self.vertices and
				to in self.vertices[frm].edges and
				self.vertices[frm].edges[to].directed == '1'):
			return self
		if self.eid_auto_increment:
			eid = next(self.counter)
		self.vertices[frm].add_edge(eid, frm, to, elb, weight, directed='1')
		self.edges[eid] = Edge(eid, frm, to, elb, weight, directed='1')
		return self

	def undirected(self):
		add_list = []
		for e in self.edges.values():
			reid = self.find_edge_id(e.to, e.frm)
			if reid == -1:
				eid = next(self.counter)
				add_list.append([eid, e.to, e.frm, e.elb, e.weight])
				self.vertices[e.to].add_edge(eid, e.to, e.frm, e.elb, e.weight, directed='0')
			else:
				self.edges[reid].directed = '2'
				self.edges[e.eid].directed = '2'
				self.vertices[e.frm].edges[e.to].directed = '2'
				self.vertices[e.to].edges[e.frm].directed = '2'
		for (e_id, e_to, e_frm, e_elb, e_weight) in add_list:
			self.edges[e_id] = Edge(e_id, e_to, e_frm, e_elb, e_weight, directed='0')

	def display(self, directed=True):
		"""Display the graph as text."""
		display_str = ''
		print('t # {}'.format(self.gid))
		for v in self.vertices.values():
			v.display()
		for e in self.edges.values():
			if directed:
				if e.directed == '1':
					e.display()
			else:
				e.display()
		print('\n')
		return display_str
