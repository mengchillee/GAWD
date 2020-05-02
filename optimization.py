from collections import OrderedDict, Counter
import math

def get_cost(this_x, count):
	"""
	Evaluates cost of encoding integers with differential encoding
	with reference to given integer.

	INPUTS
	  this_x: Reference integer
	  count: OrderDict containing frequencies of integers

	OUTPUTS
	  cost evaluated at this_x
	"""
	this_x = int(round(this_x))
	tot = sum(count.values())
	try:
		this_card = count[this_x]
		cost = tot - this_card
	except KeyError:
		cost = tot
	for val, card in zip(count.keys(), count.values()):
		if val != this_x:
			cost += card * math.log2(abs(this_x - val))
	return cost

def dichotomous_search(x):
	"""
	Dichotomous search on cost function.

	INPUTS
	x: List containing all multiplicities (integer weights)

	OUTPUTS
	[Minimizer, Cost at minimizer]
	"""
	count = OrderedDict(sorted(Counter(x).items()))
	a = float(list(count.keys())[0])
	b = float(list(count.keys())[-1])

	while a - b > 1:
		m = int((a + b) / 2)
		if get_cost(m - 1, count) < get_cost(m + 1, count):
			b = m + 1
		else:
			a = m - 1

	cost_a = get_cost(a, count)
	cost_b = get_cost(b, count)
	if(cost_a < cost_b):
		return a, cost_a
	else:
		return b, cost_b
