from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import itertools
import time
import progressbar

from graph import AUTO_EDGE_ID
from graph import Graph
from graph import VACANT_GRAPH_ID
from graph import VACANT_VERTEX_LABEL
from optimization import dichotomous_search

import pandas as pd
import numpy as np

def eliminate_incorrect_pattern(undir_freq_subs, min_support):
	del_idx = []
	for idx1, ufs in enumerate(progressbar.progressbar(undir_freq_subs)):
		def_edge_num = len(ufs.definition.edges)
		def_vertex_num = len(ufs.definition.vertices)
		del_ins = []
		for idx2, tmp in enumerate(ufs.instances):
			ien, ivn = 0, set()
			while tmp:
				ivn.add(tmp.edge.frm)
				ivn.add(tmp.edge.to)
				ien += 1
				tmp = tmp.prev
			ivn = len(ivn)
			if ien != def_edge_num or ivn != def_vertex_num:
				del_ins.append(idx2)
		ufs.instances = list(np.delete(ufs.instances, del_ins))
		if len(ufs.instances) < min_support:
			del_idx.append(idx1)
	return np.delete(undir_freq_subs, del_idx)

def self_loop_detect(freq_subs, graphs, min_support):
	freq_subs_tmp = []
	xx = -1
	for fs in progressbar.progressbar(freq_subs):
		xx += 1
		ori_sl = []
		for e in fs.definition.edges.values():
			if e.frm == e.to:
				ori_sl.append(e.frm)
		c = np.zeros(len(fs.definition.vertices))
		for ins in fs.instances:
			if graphs[ins.gid].find_edge_directed(ins.edge.frm, ins.edge.frm) and ins.edge.frm not in ori_sl:
				c[0] += 1
			appeared = [ins.edge.frm]
			while ins:
				if ins.edge.to not in appeared:
					if graphs[ins.gid].find_edge_directed(ins.edge.to, ins.edge.to) and ins.edge.to not in ori_sl:
						c[len(appeared)] += 1
					appeared.append(ins.edge.to)
					# num += 1
				ins = ins.prev
		c = [idx for idx, cc in enumerate(c) if cc >= min_support and idx not in ori_sl]
		if len(c) == 0:
			continue
		for i in range(1, len(c) + 1):
			for j in set(itertools.combinations(c, i)):
				tmp = copy.deepcopy(fs)
				del_idx = []
				for di, ins in enumerate(tmp.instances):
					pdfs_tmp = []
					if 0 in j:
						eid = graphs[ins.gid].find_edge_id(ins.edge.frm, ins.edge.frm)
						if eid != -1:
							pdfs_tmp.append((ins.gid, graphs[ins.gid].edges[eid]))
						else:
							del_idx.append(di)
					num = 1
					prev_ins = ins
					while ins:
						if num in j:
							eid = graphs[ins.gid].find_edge_id(ins.edge.to, ins.edge.to)
							if eid != -1:
								pdfs_tmp.append((ins.gid, graphs[ins.gid].edges[eid]))
							else:
								del_idx.append(di)
						num += 1
						prev_ins = ins
						ins = ins.prev

					if di not in del_idx:
						ins = prev_ins
						for p1, p2 in pdfs_tmp:
							ins.prev = PDFS(p1, p2, None)
							ins = ins.prev
				tmp.instances = np.delete(np.array(tmp.instances), np.array(del_idx))

				if len(tmp.instances) >= min_support:
					add = []
					for idx, e in enumerate(tmp.definition.edges.values()):
						if idx == 0 and 0 in j:
							add.append(e.frm)
						if idx + 1 in j:
							add.append(e.to)
					for a in add[::-1]:
						tmp.definition.add_edge(AUTO_EDGE_ID, a, a, 1, -1)
					freq_subs_tmp.append(tmp)
	return list(np.concatenate([freq_subs, freq_subs_tmp], axis=0))

def find_directed_frequent_subgraphs(undir_freq_subs, min_support):
	dir_freq_subs = []
	for fs in progressbar.progressbar(undir_freq_subs):
		directed_arr = []
		mutual_instances = []
		for idx, ins in enumerate(fs.instances):
			directed_arr.append('')
			while ins:
				if ins.edge.directed == '2':
					mutual_instances.append(copy.deepcopy(fs.instances[idx]))
					directed_arr[-1] = directed_arr[-1] + '0'
					ins.edge.directed = '0'
				else:
					directed_arr[-1] = directed_arr[-1] + ins.edge.directed
				ins = ins.prev
		for idx, ins in enumerate(mutual_instances):
			fs.instances.append(ins)
			directed_arr.append('')
			while ins:
				if ins.edge.directed == '2':
					directed_arr[-1] = directed_arr[-1] + '1'
					ins.edge.directed = '1'
				else:
					directed_arr[-1] = directed_arr[-1] + ins.edge.directed
				ins = ins.prev
		counts = {k: v for k, v in collections.Counter(directed_arr).items() if v >= min_support}
		for k in counts.keys():
			tmp_def = copy.deepcopy(fs.definition)
			for idx, kk in enumerate(k[::-1]):
				if kk == '0':
					tmp_e = tmp_def.edges[idx]
					tmp_def.vertices[tmp_e.frm].remove_edge(tmp_e.to)
					tmp_def.vertices[tmp_e.to].add_edge(tmp_e.eid, tmp_e.to, tmp_e.frm, tmp_e.elb, tmp_e.weight, directed='1')
					tmp_def.edges[idx].frm, tmp_def.edges[idx].to = tmp_e.to, tmp_e.frm
			tmp_prj = np.array(copy.deepcopy(fs.instances))[np.where(np.array(directed_arr) == k)[0]]
			dir_freq_subs.append(Substructure(tmp_def, tmp_prj))
	return dir_freq_subs

class DFSedge(object):
	"""DFSedge class."""

	def __init__(self, frm, to, vevlb):
		"""Initialize DFSedge instance."""
		self.frm = frm
		self.to = to
		self.vevlb = vevlb

	def __eq__(self, other):
		"""Check equivalence of DFSedge."""
		return (self.frm == other.frm and
				self.to == other.to and
				self.vevlb == other.vevlb)

	def __ne__(self, other):
		"""Check if not equal."""
		return not self.__eq__(other)

	def __repr__(self):
		"""Represent DFScode in string way."""
		return '(frm={}, to={}, vevlb={})'.format(
			self.frm, self.to, self.vevlb
		)

class DFScode(list):
	"""DFScode is a list of DFSedge."""

	def __init__(self):
		"""Initialize DFScode."""
		self.rmpath = list()

	def __eq__(self, other):
		"""Check equivalence of DFScode."""
		la, lb = len(self), len(other)
		if la != lb:
			return False
		for i in range(la):
			if self[i] != other[i]:
				return False
		return True

	def __hash__(self):
		return hash(tuple(self.rmpath))

	def __ne__(self, other):
		"""Check if not equal."""
		return not self.__eq__(other)

	def __repr__(self):
		"""Represent DFScode in string way."""
		return ''.join(['[', ','.join(
			[str(dfsedge) for dfsedge in self]), ']']
		)

	def push_back(self, frm, to, vevlb):
		"""Update DFScode by adding one edge."""
		self.append(DFSedge(frm, to, vevlb))
		return self

	def to_graph(self, gid=VACANT_GRAPH_ID, is_undirected=False):
		"""Construct a graph according to the dfs code."""
		g = Graph(gid, eid_auto_increment=True)
		for dfsedge in self:
			frm, to, (vlb1, elb, vlb2) = dfsedge.frm, dfsedge.to, dfsedge.vevlb
			if vlb1 != VACANT_VERTEX_LABEL:
				g.add_vertex(frm, vlb1)
			if vlb2 != VACANT_VERTEX_LABEL:
				g.add_vertex(to, vlb2)
			g.add_edge(AUTO_EDGE_ID, frm, to, elb, -1)
		return g

	def from_graph(self, g):
		"""Build DFScode from graph `g`."""
		raise NotImplementedError('Not inplemented yet.')

	def build_rmpath(self):
		"""Build right most path."""
		self.rmpath = list()
		old_frm = None
		for i in range(len(self) - 1, -1, -1):
			dfsedge = self[i]
			frm, to = dfsedge.frm, dfsedge.to
			if frm < to and (old_frm is None or to == old_frm):
				self.rmpath.append(i)
				old_frm = frm
		return self

	def get_num_vertices(self):
		"""Return number of vertices in the corresponding graph."""
		return len(set(
			[dfsedge.frm for dfsedge in self] +
			[dfsedge.to for dfsedge in self]
		))


class PDFS(object):
	"""PDFS class."""

	def __init__(self, gid=VACANT_GRAPH_ID, edge=None, prev=None):
		"""Initialize PDFS instance."""
		self.gid = gid
		self.edge = edge
		self.prev = prev


class Projected(list):
	"""Projected is a list of PDFS.

	Each element of Projected is a projection one frequent graph in one
	original graph.
	"""

	def __init__(self):
		"""Initialize Projected instance."""
		super(Projected, self).__init__()

	def push_back(self, gid, edge, prev):
		"""Update this Projected instance."""
		self.append(PDFS(gid, edge, prev))
		return self


class History(object):
	"""History class."""

	def __init__(self, g, pdfs):
		"""Initialize History instance."""
		super(History, self).__init__()
		self.edges = list()
		self.vertices_used = collections.defaultdict(int)
		self.edges_used = collections.defaultdict(int)
		if pdfs is None:
			return
		while pdfs:
			e = pdfs.edge
			self.edges.append(e)
			(self.vertices_used[e.frm],
				self.vertices_used[e.to],
				self.edges_used[e.eid]) = 1, 1, 1

			pdfs = pdfs.prev
		self.edges = self.edges[::-1]

	def has_vertex(self, vid):
		"""Check if the vertex with vid exists in the history."""
		return self.vertices_used[vid] == 1

	def has_edge(self, eid):
		"""Check if the edge with eid exists in the history."""
		return self.edges_used[eid] == 1

class Substructure():
	def __init__(self, definition, instances):
		self.definition = definition
		self.instances = instances
		self.score = 0

	def update_gdict(self):
		self.gdict = collections.defaultdict(list)
		for i in self.instances:
			self.gdict[i.gid].append(i)

	def filter_duplicate(self, graphs):
		g_his = {}
		del_idx = set()
		for idx, ins in enumerate(self.instances):
			g = graphs[ins.gid]
			if ins.gid not in g_his.keys():
				g_his[ins.gid] = set()
			view, depth = 0, 0
			while ins:
				depth += 1
				if ((ins.edge.frm, ins.edge.to) in g_his[g.gid] or
					(ins.edge.to, ins.edge.frm) in g_his[g.gid]):
					view += 1
				g_his[g.gid].add((ins.edge.frm, ins.edge.to))
				g_his[g.gid].add((ins.edge.to, ins.edge.frm))
				ins = ins.prev
			if depth == view:
				del_idx.add(idx)
		for idx in sorted(del_idx, reverse=True):
			del self.instances[idx]

	def display(self):
		print('Instance Number: %d' % len(self.instances))
		self.definition.display()

	def count_instances(self):
		return len(set(i.gid for i in self.instances))

	def find_instances(self, gid):
		return self.gdict[gid]

	def update_edge_weight(self, graphs, method='dsearch'):
		edge_weight_sum = [[] for _ in range(len(self.definition.edges))]
		for ins in self.instances:
			num = 0
			while ins != None:
				if ins.edge.directed == '1':
					eid = graphs[ins.gid].find_edge_id(ins.edge.frm, ins.edge.to)
				elif ins.edge.directed == '0':
					eid = graphs[ins.gid].find_edge_id(ins.edge.to, ins.edge.frm)
				edge_weight_sum[num].append(graphs[ins.gid].edges[eid].weight)
				num += 1
				ins = ins.prev
		self.edge_weight = []
		for idx, i in enumerate(edge_weight_sum):
			if method == 'dsearch':
				self.definition.edges[idx].weight = dichotomous_search(i)[0]
			elif method == 'minimum':
				self.definition.edges[idx].weight = min(i)
			elif method == 'median':
				self.definition.edges[idx].weight = np.median(i)
			elif method == 'mode':
				self.definition.edges[idx].weight = max(set(i), key=i.count)
			self.edge_weight.append(self.definition.edges[idx].weight)

class gSpan(object):
	"""`gSpan` algorithm."""

	def __init__(self,
				 graphs,
				 iteration,
				 min_support=10,
				 min_num_vertices=1,
				 max_num_vertices=float('inf'),
				 max_ngraphs=float('inf'),
				 verbose=False,
				 visualize=False,
				 where=False):
		"""Initialize gSpan instance."""
		self.graphs = graphs
		self.iteration = iteration
		self._min_support = min_support
		self._min_num_vertices = min_num_vertices
		self._max_num_vertices = max_num_vertices
		self._DFScode = DFScode()
		self._support = 0
		self._frequent_size1_subgraphs = list()
		self._frequent_subgraphs = []
		self._counter = itertools.count()
		self._verbose = verbose
		self._visualize = visualize
		self._where = where
		self.timestamps = dict()
		if self._max_num_vertices < self._min_num_vertices:
			print('Max number of vertices can not be smaller than '
				  'min number of that.\n'
				  'Set max_num_vertices = min_num_vertices.')
			self._max_num_vertices = self._min_num_vertices
		self._report_df = pd.DataFrame()

	def run(self):
		"""Run the gSpan algorithm."""
		if self._max_num_vertices < 2:
			return
		root = collections.defaultdict(Projected)
		for gid, g in self.graphs.items():
			g.undirected()
			for vid, v in g.vertices.items():
				edges = self._get_forward_root_edges(g, vid)
				for e in edges:
					root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
						PDFS(gid, e, None)
					)
		for vevlb, projected in progressbar.progressbar(root.items()):
			self._DFScode.append(DFSedge(0, 1, vevlb))
			self._subgraph_mining(projected)
			self._DFScode.pop()
		return self._frequent_subgraphs

	def _get_support(self, projected):
		return len(set([pdfs.gid for pdfs in projected]))

	def _get_forward_root_edges(self, g, frm):
		result = []
		v_frm = g.vertices[frm]
		for to, e in v_frm.edges.items():
			if v_frm.vlb <= g.vertices[to].vlb:
				result.append(e)
		return result

	def _get_backward_edge(self, g, e1, e2, history):
		if e1 == e2:
			return None
		for to, e in g.vertices[e2.to].edges.items():
			if history.has_edge(e.eid) or e.to != e1.frm:
				continue
			# if reture here, then self._DFScodep[0] != dfs_code_min[0]
			# should be checked in _is_min(). or:
			if e1.elb < e.elb or (
					e1.elb == e.elb and
					g.vertices[e1.to].vlb <= g.vertices[e2.to].vlb):
				return e
		return None

	def _get_forward_pure_edges(self, g, rm_edge, min_vlb, history):
		result = []
		for to, e in g.vertices[rm_edge.to].edges.items():
			if min_vlb <= g.vertices[e.to].vlb and (not history.has_vertex(e.to)):
				result.append(e)
		return result

	def _get_forward_rmpath_edges(self, g, rm_edge, min_vlb, history):
		result = []
		to_vlb = g.vertices[rm_edge.to].vlb
		for to, e in g.vertices[rm_edge.frm].edges.items():
			new_to_vlb = g.vertices[to].vlb
			if (rm_edge.to == e.to or
					min_vlb > new_to_vlb or
					history.has_vertex(e.to)):
				continue
			if rm_edge.elb < e.elb or (rm_edge.elb == e.elb and
									   to_vlb <= new_to_vlb):
				result.append(e)
		return result

	def _is_min(self):
		if self._verbose:
			print('is_min: checking {}'.format(self._DFScode))
		if len(self._DFScode) == 1:
			return True
		g = self._DFScode.to_graph(gid=VACANT_GRAPH_ID)
		dfs_code_min = DFScode()
		root = collections.defaultdict(Projected)
		for vid, v in g.vertices.items():
			edges = self._get_forward_root_edges(g, vid)
			for e in edges:
				root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
					PDFS(g.gid, e, None))
		min_vevlb = min(root.keys())
		dfs_code_min.append(DFSedge(0, 1, min_vevlb))
		# No need to check if is min code because of pruning in get_*_edge*.

		def project_is_min(projected):
			dfs_code_min.build_rmpath()
			rmpath = dfs_code_min.rmpath
			min_vlb = dfs_code_min[0].vevlb[0]
			maxtoc = dfs_code_min[rmpath[0]].to

			backward_root = collections.defaultdict(Projected)
			flag, newto = False, 0,
			end = 0
			for i in range(len(rmpath) - 1, end, -1):
				if flag:
					break
				for p in projected:
					history = History(g, p)
					e = self._get_backward_edge(g,
												history.edges[rmpath[i]],
												history.edges[rmpath[0]],
												history)
					if e is not None:
						backward_root[e.elb].append(PDFS(g.gid, e, p))
						newto = dfs_code_min[rmpath[i]].frm
						flag = True
			if flag:
				backward_min_elb = min(backward_root.keys())
				dfs_code_min.append(DFSedge(
					maxtoc, newto,
					(VACANT_VERTEX_LABEL,
					 backward_min_elb,
					 VACANT_VERTEX_LABEL)
				))
				idx = len(dfs_code_min) - 1
				if self._DFScode[idx] != dfs_code_min[idx]:
					return False
				return project_is_min(backward_root[backward_min_elb])

			forward_root = collections.defaultdict(Projected)
			flag, newfrm = False, 0
			for p in projected:
				history = History(g, p)
				edges = self._get_forward_pure_edges(g,
													 history.edges[rmpath[0]],
													 min_vlb,
													 history)
				if len(edges) > 0:
					flag = True
					newfrm = maxtoc
					for e in edges:
						forward_root[
							(e.elb, g.vertices[e.to].vlb)
						].append(PDFS(g.gid, e, p))
			for rmpath_i in rmpath:
				if flag:
					break
				for p in projected:
					history = History(g, p)
					edges = self._get_forward_rmpath_edges(g,
														   history.edges[
															   rmpath_i],
														   min_vlb,
														   history)
					if len(edges) > 0:
						flag = True
						newfrm = dfs_code_min[rmpath_i].frm
						for e in edges:
							forward_root[
								(e.elb, g.vertices[e.to].vlb)
							].append(PDFS(g.gid, e, p))

			if not flag:
				return True

			forward_min_evlb = min(forward_root.keys())
			dfs_code_min.append(DFSedge(
				newfrm, maxtoc + 1,
				(VACANT_VERTEX_LABEL, forward_min_evlb[0], forward_min_evlb[1]))
			)
			idx = len(dfs_code_min) - 1
			if self._DFScode[idx] != dfs_code_min[idx]:
				return False
			return project_is_min(forward_root[forward_min_evlb])

		res = project_is_min(root[min_vevlb])
		return res

	def _subgraph_mining(self, projected):
		self._support = self._get_support(projected)
		if self._support < self._min_support:
			return
		if self._DFScode.get_num_vertices() >= self._min_num_vertices:
			self._frequent_subgraphs.append(Substructure(copy.deepcopy(self._DFScode).to_graph(), projected))
			self._frequent_subgraphs[-1].filter_duplicate(self.graphs)

		num_vertices = self._DFScode.get_num_vertices()
		self._DFScode.build_rmpath()
		rmpath = self._DFScode.rmpath
		maxtoc = self._DFScode[rmpath[0]].to
		min_vlb = self._DFScode[0].vevlb[0]

		forward_root = collections.defaultdict(Projected)
		backward_root = collections.defaultdict(Projected)
		for p in projected:
			g = self.graphs[p.gid]
			history = History(g, p)
			# backward
			for rmpath_i in rmpath[::-1]:
				e = self._get_backward_edge(g,
											history.edges[rmpath_i],
											history.edges[rmpath[0]],
											history)
				if e is not None:
					backward_root[
						(self._DFScode[rmpath_i].frm, e.elb)
					].append(PDFS(g.gid, e, p))
			# pure forward
			if num_vertices >= self._max_num_vertices:
				continue
			edges = self._get_forward_pure_edges(g,
												 history.edges[rmpath[0]],
												 min_vlb,
												 history)
			for e in edges:
				forward_root[
					(maxtoc, e.elb, g.vertices[e.to].vlb)
				].append(PDFS(g.gid, e, p))

			# rmpath forward
			for rmpath_i in rmpath:
				edges = self._get_forward_rmpath_edges(g,
													   history.edges[rmpath_i],
													   min_vlb,
													   history)
				for e in edges:
					forward_root[
						(self._DFScode[rmpath_i].frm,
						 e.elb, g.vertices[e.to].vlb)
					].append(PDFS(g.gid, e, p))

		# backward
		for to, elb in backward_root:
			self._DFScode.append(DFSedge(
				maxtoc, to,
				(VACANT_VERTEX_LABEL, elb, VACANT_VERTEX_LABEL))
			)
			self._subgraph_mining(backward_root[(to, elb)])
			self._DFScode.pop()
		# forward
		# No need to check if num_vertices >= self._max_num_vertices.
		# Because forward_root has no element.
		for frm, elb, vlb2 in forward_root:
			self._DFScode.append(DFSedge(
				frm, maxtoc + 1,
				(VACANT_VERTEX_LABEL, elb, vlb2))
			)
			self._subgraph_mining(forward_root[(frm, elb, vlb2)])
			self._DFScode.pop()

		return self
