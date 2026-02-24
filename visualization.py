import matplotlib.pyplot as plt

def draw_line_plot(x_axis, y_axis, x_label, y_label, title):
    plt.figure(figsize=(9, 6))

    plt.plot(x_axis, y_axis, marker="o", linestyle="-", linewidth=2, markersize=6)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # make it looks nicer
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(True, linestyle="--", alpha=0.5, color="#B0BEC5")
    plt.tight_layout()
    plt.show()

def draw_multi_line_plot(x_axis, y_data_dict, x_label, y_label, title):
    plt.figure(figsize=(9, 6))

    # up to 8 y_data
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    colors = ['#2A9D8F', '#E76F51', '#3D5A80', '#F4A261', '#9B5DE5', '#00BBF9', '#F15BB5', '#00F5D4']

    for (algo_name, y_axis), marker, color in zip(y_data_dict.items(), markers, colors):
        plt.plot(x_axis, y_axis, marker=marker, color=color, linestyle="-",
                 linewidth=2.5, markersize=8, label=algo_name)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # make it looks nicer
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xticks(x_axis)
    plt.legend(frameon=True, fancybox=True, shadow=True, borderpad=1, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5, color="#B0BEC5")
    plt.tight_layout()
    plt.show()