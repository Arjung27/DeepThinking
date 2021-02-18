""" generate_dataset.py
    For generating maze dataset for the
    DeepThinking project.
    July 2020
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from create_maze import create_maze
from dijikstra import Node, path_search_algo, find_path

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


def gen_sample(num, ix, iy, ex, ey):
    """Returns a numpy array corresponding to an nxn maze and the length
     of the shortest path from (ix,iy) to (ex,ey)"""
    maze = create_maze(num, ix, iy)
    start_node = Node(2 * ix + 1, 2 * iy + 1, 0.0, -1)
    end_node = Node(2 * ex + 1, 2 * ey + 1, 0.0, -1)
    path_search_algo(start_node, end_node, maze[:, :, 0], maze.shape[0], maze.shape[1])
    coords = find_path(end_node)
    solution = np.zeros((maze.shape[0], maze.shape[1], 1))
    solution[coords[1, :], coords[0, :]] = 1
    return maze, len(coords[0]) - 1, solution


def get_final_maze_array(arr, ix, iy, ex, ey):
    """Add the start and end points to a maze and cast as uint8"""
    maze_array = arr.copy()
    maze_array[2 * iy + 1, 2 * ix + 1, :] = [0, 1, 0]
    maze_array[2 * ey + 1, 2 * ex + 1, :] = [1, 0, 0]
    return maze_array


def gen_dataset(num_images=60000, maze_size=7):
    """Function to generate a whole dataset"""
    num_images = int(num_images)
    data_array = np.zeros((num_images, 2 * maze_size + 1, 2 * maze_size + 1, 3))
    targets_array = np.zeros(num_images)
    solution_array = np.zeros((num_images, 2 * maze_size + 1, 2 * maze_size + 1, 1))
    start_and_end_array = np.zeros((num_images, 4))
    x_points, y_points = np.meshgrid(np.arange(0, maze_size), np.arange(0, maze_size))
    x_points = x_points.flatten()
    y_points = y_points.flatten()
    for j in range(num_images):
        start, end = np.random.choice(maze_size ** 2, 2, replace=False)
        ix = x_points[start]
        iy = y_points[start]
        ex = x_points[end]
        ey = y_points[end]
        maze, length, solution = gen_sample(maze_size, ix, iy, ex, ey)
        maze_array = get_final_maze_array(maze, ix, iy, ex, ey)
        data_array[j] = maze_array
        targets_array[j] = length
        solution_array[j] = solution
        start_and_end_array[j] = [ix, iy, ex, ey]
        if j % 5000 == 0:
            print(j)
    # big_data_array = np.zeros((num_images, 4 * maze_size + 4, 4 * maze_size + 4, 3))
    # big_data_array[:, 1:-1:2, 1:-1:2, :] = data_array
    # big_data_array[:, 2::2, 2::2, :] = data_array
    # big_data_array[:, 2::2, 1:-1:2, :] = data_array
    # big_data_array[:, 1:-1:2, 2::2, :] = data_array

    border = (32 - 4 * maze_size) // 2
    big_data_array = np.zeros((num_images, 32, 32, 3))
    big_data_array[:, border-1:-border:2, border-1:-border:2, :] = data_array
    big_data_array[:, border:-border+1:2, border:-border+1:2, :] = data_array
    big_data_array[:, border:-border+1:2, border-1:-border:2, :] = data_array
    big_data_array[:, border-1:-border:2, border:-border+1:2, :] = data_array

    big_solution_array = np.zeros((num_images, 32, 32, 3))
    big_solution_array[:, border-1:-border:2, border-1:-border:2, :] = solution_array
    big_solution_array[:, border:-border+1:2, border:-border+1:2, :] = solution_array
    big_solution_array[:, border:-border+1:2, border-1:-border:2, :] = solution_array
    big_solution_array[:, border-1:-border:2, border:-border+1:2, :] = solution_array
    return big_data_array, targets_array, start_and_end_array, big_solution_array


if __name__ == "__main__":
    maze_sizes = [7]
    dataset_size_multipliers = [1]
    # maze_sizes = [4, 5, 7, 9, 15, 21]
    # dataset_size_multipliers = [1, 1, 3, 6, 12, 15]

    for size, factor in zip(maze_sizes, dataset_size_multipliers):
        print(size, factor)
        for i, data_name in enumerate([f"train_{size}", f"test_{size}"]):
            # pass
            inputs, targets, start_and_end, solutions = gen_dataset([50000 * factor, 10000 * factor][i], size)
            # inputs, targets, start_and_end, solutions = gen_dataset([5, 1][i], size)
            unique, frequency = np.unique(targets, return_counts=True)
            fig, ax = plt.subplots()
            ax.hist(targets, bins=len(unique))
            plt.savefig(f"historgram_of_labels{data_name}.pdf")
            if not os.path.isdir(f"{data_name}"):
                os.makedirs(f"{data_name}")
            np.save(os.path.join(data_name, "inputs.npy"), inputs)
            np.save(os.path.join(data_name, "targets.npy"), targets)
            np.save(os.path.join(data_name, "start_and_end.npy"), start_and_end)
            np.save(os.path.join(data_name, "solutions.npy"), solutions)

    # train_path, test_path = ("train_5", "test_5")
    # train_inputs_np = np.load(os.path.join(train_path, "inputs.npy"))
    # train_targets_np = np.load(os.path.join(train_path, "solutions.npy"))
    # test_inputs_np = np.load(os.path.join(test_path, "inputs.npy"))
    # test_targets_np = np.load(os.path.join(test_path, "solutions.npy"))
    #
    # repeats = 0
    # i = 0
    # for tp in test_inputs_np:
    #     i += 1
    #     for ti in train_inputs_np:
    #         if (tp == ti).all():
    #             repeats += 1
    #             continue
    #     if i % 10 == 0:
    #         print(i, repeats)
    # print(f"There are {repeats} mazes in the testset that also appear in the trainset.")