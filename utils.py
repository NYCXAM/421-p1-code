import math
import sys
from random import random

# parse the plain text into matrix
def parse_graph(filename):
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            node = []
            for val in line.split():
                node.append(float(val))
            matrix.append(node)

    return matrix

# return a random node as an index within size range
def random_node(size):
    return round(random() * size)

def visit(curr_node, visited, graph):
    raise NotImplementedError

# Nearest Neighbor
def NN(filename):
    matrix = parse_graph(filename)
    visited, path = [], []
    curr_node = random_node(len(matrix))
    visited.append(curr_node)
    path.append(curr_node)

    while len(visited) != len(matrix):
        min_val = 0
        for i in range(len(matrix)):
            if i not in visited or i != curr_node:
                if matrix[curr_node][i] < matrix[curr_node][min_val]:
                    min_val = i

            curr_node = i
            visited.append(curr_node)
            path.append(curr_node)
    path.append(path[0])
    return path

# NN with 2-Opt
def NN_2opt():
    raise NotImplementedError

# Repeated Random NN
def RRNN():
    raise NotImplementedError
