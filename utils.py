import math
import os
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

def path_length(path, matrix):
    length = 0
    for i in range(len(path) - 2):
        length += matrix[path[i]][path[i + 1]]

    return length

def random_matrices(size):
    all_matrices = []
    for m in os.listdir('\matrices'):
        if os.path.isfile(os.paht.join('matrices', m)):
            all_matrices.append(parse_graph('matrices/' + m))
    return random.sample(all_matrices, size)


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
def NN_2opt(filename):
    matrix = parse_graph(filename)
    path = NN(filename)
    better_path = True
    while better_path:
        better_path = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                old_dist = matrix[path[i - 1]][path[i]] + matrix[path[j]][path[j + 1]]
                new_dist = matrix[path[i - 1]][path[j]] + matrix[path[i]][path[j + 1]]
                if new_dist < old_dist:
                    path[i:j+1] = reversed(path[i:j+1])
                    better_path = True
                    break
            if better_path:
                break
    return path


# Repeated Random NN
def RRNN(filename, k, num_repeats):
    matrix = parse_graph(filename)
    path = []
    best_dist = float('inf')

    for _ in range(num_repeats):
        # random nearest neighbor part
        curr_node = random_node(len(matrix))
        unvisited, curr_path = [], []

        unvisited.append(curr_node)
        curr_path.append(curr_node)

        while unvisited:
            neighbors = []
            for node in unvisited:
                neighbors.append((matrix[curr_node][node], node))

            neighbors.sort(key=lambda x: x[0])
            limit = min(k, len(neighbors))
            next_node = random_node(neighbors[:limit])[1]

            path.append(next_node)
            unvisited.remove(next_node)
            curr_node = next_node

        path.append(path[0])

        # 2opt part
        better_path = True
        while better_path:
            better_path = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):
                    old_dist = matrix[path[i - 1]][path[i]] + matrix[path[j][path[j + 1]]]
                    new_dist = matrix[path[i - 1]][path[j]] + matrix[path[j][path[j + 1]]]

                    if new_dist < old_dist:
                        path[i:j+1] = reversed(path[i:j+1])
                        better_path = True
                        break
                if better_path:
                    break

        curr_dist = path_length(path, matrix)
        if curr_dist < best_dist:
            best_dist = curr_dist
            best_path = path
    return best_path



