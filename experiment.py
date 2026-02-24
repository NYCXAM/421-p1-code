import utils
import algorithms as algos
import json
import time
import sys
import statistics as stat
import visualization as vis

# helper function for NN / NN_2opt / RRNN
def benchmark(func, *args):
    real_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()

    path = func(*args)

    real_time = time.perf_counter_ns() - real_start
    cpu_time = time.process_time_ns() - cpu_start

    if cpu_time == 0:
        iterations = 100
        cpu_start = time.process_time_ns()
        for _ in range(iterations):
            func(*args)
        cpu_time = (time.process_time_ns() - cpu_start) / iterations

    filename = args[0]
    cost = utils.get_cost(path, filename, None)
    return real_time, cpu_time, cost

# helper function for a*
def benchmark_astar(filename):
    real_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()

    path, cost, nodes_expanded = algos.Astar(filename)

    real_time = time.perf_counter_ns() - real_start
    cpu_time = time.process_time_ns() - cpu_start

    if cpu_time == 0:
        iterations = 100
        cpu_start = time.process_time_ns()
        for _ in range(iterations):
            algos.Astar(filename)
        cpu_time = (time.process_time_ns() - cpu_start) / iterations

    return real_time, cpu_time, cost, nodes_expanded

# Part 1
def part1_1(matrices):
    parsed_lengths = {m: len(utils.parse_matrix(m)) for m in matrices}
    largest_size = max(parsed_lengths.values())
    target_matrices = [m for m in matrices if parsed_lengths[m] == largest_size]

    k_val = [1, 2, 3, 5, 7, 10]
    num_repeats = [1, 5, 10, 15, 30, 50]

    results = {
        "k_test": {"k_values": k_val, "cost": []},
        "repeats_test": {"num_repeats_values": num_repeats, "cost": []},
        "target_size": largest_size
    }

    # test k values, num_repeats = 30
    for k in k_val:
        cost = []
        for m in target_matrices:
            path = algos.RRNN(m, k, 30)
            cost.append(utils.get_cost(path, m, None))
        results["k_test"]["cost"].append(stat.median(cost))

    # test num_repeats, k = 5
    for num in num_repeats:
        cost = []
        for m in target_matrices:
            path = algos.RRNN(m, 5, num)
            cost.append(utils.get_cost(path, m, None))
        results["repeats_test"]["cost"].append(stat.median(cost))

    return results

def part1_2(matrices):
    # group by matrix size
    raw_data = {}

    for m in matrices:
        matrix = utils.parse_matrix(m)
        size = len(matrix)
        if size not in raw_data:
            raw_data[size] = {
                "NN": {"real": [], "cpu": [], "cost": []},
                "NN_2opt": {"real": [], "cpu": [], "cost": []},
                "RRNN": {"real": [], "cpu": [], "cost": []}
            }

        metrics = {
            "NN": benchmark(algos.NN, m),
            "NN_2opt": benchmark(algos.NN_2opt, m),
            "RRNN": benchmark(algos.RRNN, m, 5, 30)
        }

        for algo, (r_time, c_time, cost) in metrics.items():
            raw_data[size][algo]["real"].append(r_time)
            raw_data[size][algo]["cpu"].append(c_time)
            raw_data[size][algo]["cost"].append(cost)

    # Calculate medians for plotting
    sorted_sizes = sorted(raw_data.keys())

    plot_data = {
        "sizes": sorted_sizes,
        "real_time": {"NN": [], "NN_2opt": [], "RRNN": []},
        "cpu_time": {"NN": [], "NN_2opt": [], "RRNN": []},
        "cost": {"NN": [], "NN_2opt": [], "RRNN": []}
    }

    for size in sorted_sizes:
        for algo in ["NN", "NN_2opt", "RRNN"]:
            plot_data["real_time"][algo].append(stat.median(raw_data[size][algo]["real"]))
            plot_data["cpu_time"][algo].append(stat.median(raw_data[size][algo]["cpu"]))
            plot_data["cost"][algo].append(stat.median(raw_data[size][algo]["cost"]))

    return plot_data

# helper function to get median mqtrices per size
def compute_astar_stats(matrices):
    astar_raw = {}
    for filename in matrices:
        matrix = utils.parse_matrix(filename)
        size = len(matrix)
        if size not in astar_raw:
            astar_raw[size] = {"real": [], "cpu": [], "cost": [], "nodes": []}

        real_time, cpu_time, cost, nodes = benchmark_astar(filename)
        astar_raw[size]["real"].append(real_time)
        astar_raw[size]["cpu"].append(cpu_time)
        astar_raw[size]["cost"].append(cost)
        astar_raw[size]["nodes"].append(nodes)

    sorted_sizes = sorted(astar_raw.keys())
    astar_medians = {
        "sizes": sorted_sizes,
        "real_time": [],
        "cpu_time": [],
        "cost": [],
        "nodes_expanded": [],
    }
    for size in sorted_sizes:
        data = astar_raw[size]
        astar_medians["real_time"].append(stat.median(data["real"]))
        astar_medians["cpu_time"].append(stat.median(data["cpu"]))
        astar_medians["cost"].append(stat.median(data["cost"]))
        astar_medians["nodes_expanded"].append(stat.median(data["nodes"]))

    return astar_medians


def part2_1(base_stats, astar_stats):
    sizes = base_stats["sizes"]

    ratio_real = {"NN": [], "NN_2opt": [], "RRNN": []}
    ratio_cpu = {"NN": [], "NN_2opt": [], "RRNN": []}
    ratio_cost = {"NN": [], "NN_2opt": [], "RRNN": []}

    for i, size in enumerate(sizes):
        a_real = astar_stats["real_time"][i]
        a_cpu = astar_stats["cpu_time"][i]
        a_cost = astar_stats["cost"][i]
        for algo in ["NN", "NN_2opt", "RRNN"]:
            ratio_real[algo].append(
                base_stats["real_time"][algo][i] / a_real if a_real > 0 else float("nan")
            )
            ratio_cpu[algo].append(
                base_stats["cpu_time"][algo][i] / a_cpu if a_cpu > 0 else float("nan")
            )
            ratio_cost[algo].append(
                base_stats["cost"][algo][i] / a_cost if a_cost > 0 else float("nan")
            )

    return {
        "sizes": sizes,
        "real_time_ratio": ratio_real,
        "cpu_time_ratio": ratio_cpu,
        "cost_ratio": ratio_cost,
    }


def part2_2(astar_stats):
    return {
        "sizes": astar_stats["sizes"],
        "nodes_expanded": astar_stats["nodes_expanded"],
    }


def part3_1(matrices):
    results = {
        "HC": {"restarts": [1, 5, 10, 20, 50], "costs": []},
        "SA": {"alphas": [0.8, 0.9, 0.95, 0.99], "costs": []},
        "GA": {"mutations": [0.01, 0.05, 0.1, 0.2, 0.5], "costs": []}
    }

    for r in results["HC"]["restarts"]:
        costs = [utils.get_cost(algos.hill_climbing(m, r), m, None) for m in matrices]
        results["HC"]["costs"].append(stat.median(costs))

    for a in results["SA"]["alphas"]:
        costs = [utils.get_cost(algos.simulated_annealing(m, a, 1000, 1000), m, None) for m in matrices]
        results["SA"]["costs"].append(stat.median(costs))

    for mc in results["GA"]["mutations"]:
        costs = [utils.get_cost(algos.genetic_algorithm(m, mc, 20, 50), m, None) for m in matrices]
        results["GA"]["costs"].append(stat.median(costs))

    return results


def part3_2(single_matrix):
    return {
        "HC_history": algos.hill_climbing(single_matrix, 50, track_history=True)[1],
        "SA_history": algos.simulated_annealing(single_matrix, 0.95, 1000, 500, track_history=True)[1],
        "GA_history": algos.genetic_algorithm(single_matrix, 0.1, 20, 50, track_history=True)[1]
    }


def part3_3(matrices, astar_stats):
    raw_data = {}
    for m in matrices:
        size = len(utils.parse_matrix(m))
        if size not in raw_data:
            raw_data[size] = {
                "HC": {"real": [], "cpu": [], "cost": []},
                "SA": {"real": [], "cpu": [], "cost": []},
                "GA": {"real": [], "cpu": [], "cost": []}
            }

        metrics = {
            "HC": benchmark(algos.hill_climbing, m, 10),
            "SA": benchmark(algos.simulated_annealing, m, 0.95, 1000, 1000),
            "GA": benchmark(algos.genetic_algorithm, m, 0.1, 20, 50)
        }

        for algo, (r_time, c_time, cost) in metrics.items():
            raw_data[size][algo]["real"].append(r_time)
            raw_data[size][algo]["cpu"].append(c_time)
            raw_data[size][algo]["cost"].append(cost)

    sizes = sorted(raw_data.keys())
    ratio_real, ratio_cpu, ratio_cost = {}, {}, {}

    for algo in ["HC", "SA", "GA"]:
        ratio_real[algo], ratio_cpu[algo], ratio_cost[algo] = [], [], []
        for i, size in enumerate(sizes):
            a_real = astar_stats["real_time"][i]
            a_cpu = astar_stats["cpu_time"][i]
            a_cost = astar_stats["cost"][i]

            ratio_real[algo].append(stat.median(raw_data[size][algo]["real"]) / a_real if a_real > 0 else float("nan"))
            ratio_cpu[algo].append(stat.median(raw_data[size][algo]["cpu"]) / a_cpu if a_cpu > 0 else float("nan"))
            ratio_cost[algo].append(stat.median(raw_data[size][algo]["cost"]) / a_cost if a_cost > 0 else float("nan"))

    return {"sizes": sizes, "real_time_ratio": ratio_real, "cpu_time_ratio": ratio_cpu, "cost_ratio": ratio_cost}


def main():
    # handle command line
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(f"--- Evaluating Single File: {filename} ---")

        # run benchmarks on the provided file
        metrics = {
            "NN": benchmark(algos.NN, filename),
            "NN_2opt": benchmark(algos.NN_2opt, filename),
            "RRNN": benchmark(algos.RRNN, filename, 5, 30),
            "HC": benchmark(algos.hill_climbing, filename, 10),
            "SA": benchmark(algos.simulated_annealing, filename, 0.95, 1000, 1000),
            "GA": benchmark(algos.genetic_algorithm, filename, 0.1, 20, 50)
        }

        for algo_name, (r_time, c_time, cost) in metrics.items():
            print(f"{algo_name.ljust(8)} | Cost: {cost:<8.2f} | CPU Time: {c_time} ns")

        # exit so the full experiment don't run
        return
    # ----------------------------------------------
    # Part 1
    print("\nRunning Part 1...")
    matrices = utils.random_matrices(3)
    part1_1_result = part1_1(matrices)
    print(json.dumps(part1_1_result))

    vis.draw_line_plot(
        part1_1_result["k_test"]["k_values"],
        part1_1_result["k_test"]["cost"],
        "k values",
        "Costs",
        "(RRNN) Median Cost vs k_values",
    )
    vis.draw_line_plot(
        part1_1_result["repeats_test"]["num_repeats_values"],
        part1_1_result["repeats_test"]["cost"],
        "num repeats",
        "Costs",
        "(RRNN) Median Cost vs num_repeats",
    )

    part1_2_result = part1_2(matrices)
    print(json.dumps(part1_2_result))
    sizes = part1_2_result["sizes"]

    # Part 1.2
    vis.draw_multi_line_plot(
        sizes,
        part1_2_result["real_time"],
        "Matrix Sizes",
        "Total Real Time (ns)",
        "Median Real Runtime vs Matrix Sizes",
    )

    vis.draw_multi_line_plot(
        sizes,
        part1_2_result["cpu_time"],
        "Matrix Sizes",
        "CPU Time (ns)",
        "Median CPU Time vs Matrix Sizes",
    )

    vis.draw_multi_line_plot(
        sizes,
        part1_2_result["cost"],
        "Matrix Sizes",
        "Cost (Total Distance)",
        "Median Path Cost vs Matrix Sizes",
    )

    # ----------------------------------------------
    # Part 2
    print("\nRunning Part 2...")
    # use only sizes 5, 6, 7, 10, 15 so A* runs faster
    PART2_SIZE_GROUPS = (5, 6, 7, 10, 15)
    matrices_part2 = utils.random_matrices(5, PART2_SIZE_GROUPS)
    part1_2_result_part2 = part1_2(matrices_part2)
    astar_stats = compute_astar_stats(matrices_part2)

    # Part 2.1
    part2_1_result = part2_1(part1_2_result_part2, astar_stats)
    print(json.dumps(part2_1_result))

    sizes_p2 = part2_1_result["sizes"]

    vis.draw_multi_line_plot(
        sizes_p2,
        part2_1_result["real_time_ratio"],
        "Matrix sizes",
        "Real time / A* real time",
        "Real time vs A* (normalized)",
    )
    vis.draw_multi_line_plot(
        sizes_p2,
        part2_1_result["cpu_time_ratio"],
        "Matrix sizes",
        "CPU time / A* CPU time",
        "CPU time vs A* (normalized)",
    )
    vis.draw_multi_line_plot(
        sizes_p2,
        part2_1_result["cost_ratio"],
        "Matrix sizes",
        "Cost / A* cost",
        "Path cost vs A* (normalized)",
    )

    # Part 2.2
    part2_2_result = part2_2(astar_stats)
    print(json.dumps(part2_2_result))

    vis.draw_line_plot(
        part2_2_result["sizes"],
        part2_2_result["nodes_expanded"],
        "Matrix sizes",
        "Median nodes expanded by A*",
        "Median A* nodes expanded vs matrix size",
    )


    # ----------------------------------------------
    # Part 3
    print("\nRunning Part 3...")
    # Part 3.1
    p3_1_res = part3_1(matrices_part2)  # Run on smaller subset to save time
    vis.draw_line_plot(
        p3_1_res["HC"]["restarts"],
        p3_1_res["HC"]["costs"],
        "Number of Restarts",
        "Median Cost",
        "(HC) Cost vs Restarts"
    )
    vis.draw_line_plot(
        p3_1_res["SA"]["alphas"],
        p3_1_res["SA"]["costs"],
        "Cooling Alpha",
        "Median Cost",
        "(SA) Cost vs Alpha"
    )
    vis.draw_line_plot(
        p3_1_res["GA"]["mutations"],
        p3_1_res["GA"]["costs"],
        "Mutation Chance",
        "Median Cost",
        "(GA) Cost vs Mutation Chance"
    )

    # Part 3.2
    target_matrix = matrices_part2[-1]
    p3_2_res = part3_2(target_matrix)

    vis.draw_line_plot(
        list(range(len(p3_2_res["HC_history"]))),
        p3_2_res["HC_history"],
        "Restart Iteration",
        "Best Cost",
        "(HC) Cost Improvement over Restarts"
    )
    vis.draw_line_plot(
        list(range(len(p3_2_res["SA_history"]))),
        p3_2_res["SA_history"],
        "Iteration",
        "Best Cost",
        "(SA) Cost Improvement over Iterations"
    )
    vis.draw_line_plot(
        list(range(len(p3_2_res["GA_history"]))),
        p3_2_res["GA_history"],
        "Generation",
        "Best Cost",
        "(GA) Cost Improvement over Generations"
    )

    # Part 3.3
    p3_3_res = part3_3(matrices_part2, astar_stats)

    vis.draw_multi_line_plot(
        p3_3_res["sizes"],
        p3_3_res["real_time_ratio"],
        "Matrix Sizes",
        "Real Time / A* Real Time",
        "Real Time vs A*"
    )
    vis.draw_multi_line_plot(
        p3_3_res["sizes"],
        p3_3_res["cpu_time_ratio"],
        "Matrix Sizes",
        "CPU Time / A* CPU Time",
        "CPU Time vs A*"
    )
    vis.draw_multi_line_plot(
        p3_3_res["sizes"],
        p3_3_res["cost_ratio"],
        "Matrix Sizes",
        "Cost / A* Cost",
        "Path Cost vs A*"
    )
    print("\nAll Experiments Finished")

if __name__ == "__main__":
    main()