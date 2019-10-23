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

def dichotomous_search(x, tol = 0.1):
    """
    Dichotomous search on cost function.

    INPUTS
        x: List containing all multiplicities (integer weights)
        get_cost(): Function to use for evaluating cost

    OUTPUTS
        [Minimizer, Cost at minimizer]

    """
    count = OrderedDict(sorted(Counter(x).items()))
    a = float(list(count.keys())[0])
    b = float(list(count.keys())[-1])
    cost_a = get_cost(a,count)
    cost_b = get_cost(b,count)

    while abs(a - b) >= tol:
        if cost_a < cost_b:
            b = a + abs(a-b)*0.5
            cost_b = get_cost(b, count)
        else:
            a = b - abs(a-b)*0.5
            cost_a = get_cost(a, count)

    return int(round((a + b) / 2.0)), get_cost((a + b) / 2.0, count)
