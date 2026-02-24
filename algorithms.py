import utils
import random
import heapq
import math
import numpy as np
import itertools
import scipy.sparse.csgraph as csgraph

# Nearest Neighbor
def NN(filename):
    matrix = utils.parse_matrix(filename)
    visited, path = [], []
    curr_node = utils.random_node(len(matrix))
    visited.append(curr_node)
    path.append(curr_node)

    while len(visited) != len(matrix):
        min_val = None
        for i in range(len(matrix)):
            if i not in visited:
                if min_val is None or matrix[curr_node][i] < matrix[curr_node][min_val]:
                    min_val = i

        curr_node = min_val
        visited.append(curr_node)
        path.append(curr_node)

    path.append(path[0])
    return path

# NN with 2-Opt
def NN_2opt(filename):
    matrix = utils.parse_matrix(filename)
    path = NN(filename)
    better_path = True

    while better_path:
        better_path = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                # compute dist before / after swap
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
    matrix = utils.parse_matrix(filename)
    best_path = []
    best_dist = float('inf')

    for _ in range(num_repeats):
        # random nearest neighbor part
        curr_node = utils.random_node(len(matrix))
        unvisited = list(range(len(matrix)))
        path = []

        path.append(curr_node)
        unvisited.remove(curr_node)

        while unvisited:
            neighbors = []
            for node in unvisited:
                neighbors.append((matrix[curr_node][node], node))

            neighbors.sort(key=lambda x: x[0])
            limit = min(k, len(neighbors))
            next_node = random.choice(neighbors[:limit])[1]

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
                    old_dist = matrix[path[i - 1]][path[i]] + matrix[path[j]][path[j + 1]]
                    new_dist = matrix[path[i - 1]][path[j]] + matrix[path[i]][path[j + 1]]

                    if new_dist < old_dist:
                        path[i:j+1] = reversed(path[i:j+1])
                        better_path = True
                        break
                if better_path:
                    break

        curr_dist = utils.get_cost(path, None, matrix)
        if curr_dist < best_dist:
            best_dist = curr_dist
            best_path = path

    return best_path

# helper for A*
def MST_heuristic(matrix, curr_node, unvisited):
    np_matrix = np.array(matrix)

    target_nodes = [curr_node] + list(unvisited)
    sub_matrix = np_matrix[np.ix_(target_nodes, target_nodes)]

    mst = csgraph.minimum_spanning_tree(sub_matrix)
    return mst.sum()


def Astar(filename):
    matrix = utils.parse_matrix(filename)
    start_node = 0

    # use frozenset for faster speed
    unvisited = frozenset(range(1, len(matrix)))

    nodes_expanded = 0
    h_init = MST_heuristic(matrix, start_node, unvisited)
    nodes_expanded += 1

    tie_breaker = itertools.count()

    #(f_score, tie_break, g_score, curr_node, unvisited, path)
    state_init = (h_init, next(tie_breaker), 0, start_node, unvisited, [start_node])
    min_heap = [state_init]

    # maps (curr_node, unvisited_frozenset) -> min_g_score
    best_g = {(start_node, unvisited): 0}

    while min_heap:
        f, _, g, curr_node, unvisited, path = heapq.heappop(min_heap)

        # goal state: visited all cities and returned to the start
        if not unvisited and curr_node == start_node:
            return path, g, nodes_expanded

        if g > best_g.get((curr_node, unvisited), float('inf')):
            continue

        # if unvisited is empty, our only valid move is to return to the start node
        if not unvisited:
            final_g = g + matrix[curr_node][start_node]
            state_key = (start_node, unvisited)

            if final_g < best_g.get(state_key, float('inf')):
                best_g[state_key] = final_g
                final_path = path + [start_node]
                # h=0 since no cities remain, so f = final_g
                heapq.heappush(min_heap, (final_g, next(tie_breaker), final_g, start_node, unvisited, final_path))
            continue

        for next_node in unvisited:
            new_g = g + matrix[curr_node][next_node]
            new_unvisited = unvisited - frozenset([next_node])
            state_key = (next_node, new_unvisited)

            if new_g >= best_g.get(state_key, float('inf')):
                continue

            best_g[state_key] = new_g
            new_h = MST_heuristic(matrix, next_node, new_unvisited)
            nodes_expanded += 1

            new_f = new_g + new_h
            new_path = path + [next_node]

            new_state = (new_f, next(tie_breaker), new_g, next_node, new_unvisited, new_path)
            heapq.heappush(min_heap, new_state)

    return None, float("inf"), nodes_expanded


def hill_climbing(filename, num_restarts, track_history=False):
    matrix = utils.parse_matrix(filename)
    n = len(matrix)
    global_best_path = None
    global_best_cost = float('inf')
    # change threshold with the matrix size
    max_failed_swaps = n * 50
    history = []

    for _ in range(num_restarts):
        # generate random valid tour
        nodes = list(range(1, n))
        random.shuffle(nodes)
        current_path = [0] + nodes + [0]
        current_cost = sum(matrix[current_path[k]][current_path[k + 1]] for k in range(len(current_path) - 1))

        failed_swaps = 0
        # keep climbing until find a minimum
        while failed_swaps < max_failed_swaps:
            # swap 2 nodes
            i, j = random.sample(range(1, n), 2)
            neighbor_path = current_path[:]
            neighbor_path[i], neighbor_path[j] = neighbor_path[j], neighbor_path[i]
            neighbor_cost = sum(matrix[neighbor_path[k]][neighbor_path[k + 1]] for k in range(len(neighbor_path) - 1))

            if neighbor_cost < current_cost:
                current_path = neighbor_path
                current_cost = neighbor_cost
                failed_swaps = 0
            else:
                failed_swaps += 1

        if current_cost < global_best_cost:
            global_best_cost = current_cost
            global_best_path = current_path

        if track_history:
            history.append(global_best_cost)

    if track_history:
        return global_best_path, history
    return global_best_path

def simulated_annealing(filename, alpha, initial_temperature, max_iterations, track_history=False):
    matrix = utils.parse_matrix(filename)
    n = len(matrix)
    # generate the init random solution
    nodes = list(range(1, n))
    random.shuffle(nodes)
    current_path = [0] + nodes + [0]
    current_cost = sum(matrix[current_path[k]][current_path[k + 1]] for k in range(len(current_path) - 1))

    best_path = current_path[:]
    best_cost = current_cost
    t = initial_temperature
    history = []

    for _ in range(max_iterations):
        if t <= 0:
            break

        # generate neighboring solution by swapping nodes
        i, j = random.sample(range(1, n), 2)
        neighbor_path = current_path[:]
        neighbor_path[i], neighbor_path[j] = neighbor_path[j], neighbor_path[i]
        neighbor_cost = sum(matrix[neighbor_path[k]][neighbor_path[k + 1]] for k in range(len(neighbor_path) - 1))

        accepted = False
        if neighbor_cost < current_cost:
            accepted = True
        else:
            # get the probability of choosing the bad solution
            try:
                probability = math.exp((current_cost - neighbor_cost) / t)
                if random.random() < probability:
                    accepted = True
            except OverflowError:
                pass

        # accept and cool temperature
        if accepted:
            current_path = neighbor_path
            current_cost = neighbor_cost
            t *= alpha
            if current_cost < best_cost:
                best_cost = current_cost
                best_path = current_path[:]

        if track_history:
            history.append(best_cost)

    if track_history:
        return best_path, history
    return best_path


def order_crossover(parent1, parent2):
    # get inner nodes
    inner1 = parent1[1:-1]
    inner2 = parent2[1:-1]
    size = len(inner1)

    # get random slice bounds
    start, end = sorted(random.sample(range(size), 2))
    child_inner = [None] * size

    # copy the slice from parent1
    child_inner[start:end + 1] = inner1[start:end + 1]

    # fill the rest from parent2
    p2_idx = 0
    for i in range(size):
        if child_inner[i] is None:
            while inner2[p2_idx] in child_inner:
                p2_idx += 1
            child_inner[i] = inner2[p2_idx]

    return [0] + child_inner + [0]


def genetic_algorithm(filename, mutation_chance, population_size, num_generations, track_history=False):
    matrix = utils.parse_matrix(filename)
    n = len(matrix)

    # init a random population
    population = []
    for _ in range(population_size):
        nodes = list(range(1, n))
        random.shuffle(nodes)
        path = [0] + nodes + [0]
        population.append((utils.get_cost(path, None, matrix), path))

    history = []
    for _ in range(num_generations):
        children = []
        for _ in range(population_size):
            p1, p2 = random.sample(population, 2)

            # crossover (i used ox1(
            inner1, inner2 = p1[1][1:-1], p2[1][1:-1]
            size = len(inner1)
            start, end = sorted(random.sample(range(size), 2))
            # copy the slice from parent1
            child_inner = [None] * size
            child_inner[start:end + 1] = inner1[start:end + 1]

            # fill the rest from parent2
            p2_idx = 0
            for i in range(size):
                if child_inner[i] is None:
                    while inner2[p2_idx] in child_inner:
                        p2_idx += 1
                    child_inner[i] = inner2[p2_idx]
            child_path = [0] + child_inner + [0]

            # mutation
            if random.random() < mutation_chance:
                inner = child_path[1:-1]
                if len(inner) >= 2:
                    i, j = random.sample(range(len(inner)), 2)
                    inner[i], inner[j] = inner[j], inner[i]
                child_path = [0] + inner + [0]

            children.append((utils.get_cost(child_path, None, matrix), child_path))

        # combine, sort, and slice to keep the best
        population = population + children
        population.sort(key=lambda x: x[0])
        population = population[:population_size]

        if track_history:
            history.append(population[0][0])

    if track_history:
        return population[0][1], history

    return population[0][1]
