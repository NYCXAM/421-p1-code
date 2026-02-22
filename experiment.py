import utils
import json
# Part 1
def part1_1(matrices):
    # 1. Find optimal hyperparameters for RRNN
    # k values range(1, 2, 3, 5, 7, 10, 15), num_repeats = 100
    k_val = [1, 2, 3, 5, 7, 10, 15]
    cost_k = []
    for k in k_val:
        for m in matrices:
            result = utils.RRNN(m, k, 100)
            cost_k.append(result)

    # num_repeats range(1, 10, 50, 100, 500, 1000), k value = 5
    num_repeats = [1, 10, 50, 100, 500, 1000]
    cost_n = []
    for num in num_repeats:
        for m in matrices:
            result = utils.RRNN(m, 5, num)
            cost.append(result)
    results = {
        "k_val": k_val,
        "cost_k": cost_k,
        "num_repeats": num_repeats,
        "cost_n": cost_n,
        "matrices": matrices,
    }
    return results

def part1_2(matrices):
    nn_reals, nn_cpus, nn_costs, opt_reals, opt_cpus, opt_costs, rrnn_reals, rrnn_cpus, rrnn_costs = (
        [], [], [], [], [], [], [], [], [])
    for m in matrices:

    raise NotImplementedError


def main():
    part1_1_results = part1_1(10)