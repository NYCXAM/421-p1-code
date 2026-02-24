import math
import os
import sys
import re
import random

# parse the filename into matrix
def parse_matrix(filename):
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            node = []
            for val in line.split():
                node.append(float(val))
            matrix.append(node)

    return matrix

# return a random node index in [0, size-1]
def random_node(size):
    return random.randrange(size)

# return the cost of given path
def get_cost(path, filename, matrix):
    if filename:
        matrix = parse_matrix(filename)

    length = 0
    for i in range(len(path) - 1):
        length += matrix[path[i]][path[i + 1]]

    return length

# return randomized matrices on chosen or all sizes
def random_matrices(num_matrices_per_size, size_groups=None):
    all_sizes = {}
    pattern = re.compile(r'(\d+)_random_adj_mat_\d+\.txt$')

    for filename in os.listdir("matrices"):
        match = pattern.match(filename)
        if match:
            size = int(match.group(1))
            if size_groups is not None and size not in size_groups:
                continue
            if size not in all_sizes:
                all_sizes[size] = []
            filepath = os.path.join("matrices", filename)
            all_sizes[size].append(filepath)

    file_paths = []
    for size in sorted(all_sizes.keys()):
        paths = all_sizes[size]
        n = min(num_matrices_per_size, len(paths))
        file_paths.extend(random.sample(paths, n))

    return file_paths



