""" dijkstra.py
    For solving mazes with Dijkstra's algorithm in
    order to label data for a maze dataset for the
    DeepThinking project.
    July 2020
"""
import heapq

import numpy as np

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class Node:
    """A node in the graph"""
    def __init__(self, x_coord, y_coord, cost, parentID):

        self.x = x_coord
        self.y = y_coord
        self.cost = cost
        self.parentID = parentID

    def __lt__(self, other):
        return self.cost < other.cost


def possible_steps():
    """get possible steps"""
    steps_with_cost = np.array([[0, 1, 1],              # Move_up
                                [1, 0, 1],              # Move_right
                                [0, -1, 1],             # Move_down
                                [-1, 0, 1],             # Move_left
                                ])
    return steps_with_cost


def is_valid(point_x, point_y, grid, width, height):
    """see if a point is valid"""
    if not grid[int(point_y)][int(point_x)]:
        return False
    if point_y < 0 or point_x < 0:
        return False
    if point_y > height or point_x > width:
        return False
    return True


def is_goal(current, goal):
    """see if we are at the goal"""
    return (current.x == goal.x) and (current.y == goal.y)


def path_search_algo(start_node, end_node, grid, width, height):
    """path search function"""
    current_node = start_node
    goal_node = end_node
    steps_with_cost = possible_steps()

    if is_goal(current_node, goal_node):
        return 1

    open_nodes = {}
    open_nodes[start_node.x * width + start_node.y] = start_node
    closed_nodes = {}
    cost = []
    all_nodes = []
    heapq.heappush(cost, [start_node.cost, start_node])

    while len(cost) != 0:

        current_node = heapq.heappop(cost)[1]
        all_nodes.append([current_node.x, current_node.y])
        current_id = current_node.x * width + current_node.y

        if is_goal(current_node, end_node):
            end_node.parentID = current_node.parentID
            end_node.cost = current_node.cost
            return 1, all_nodes

        if current_id in closed_nodes:
            continue
        else:
            closed_nodes[current_id] = current_node

        del open_nodes[current_id]

        for i in range(steps_with_cost.shape[0]):

            new_node = Node(current_node.x + steps_with_cost[i][0],
                            current_node.y + steps_with_cost[i][1],
                            current_node.cost + steps_with_cost[i][2],
                            current_node)

            new_node_id = new_node.x*width + new_node.y

            if not is_valid(new_node.x, new_node.y, grid, width, height):
                continue
            elif new_node_id in closed_nodes:
                continue

            if new_node_id in open_nodes:
                if new_node.cost < open_nodes[new_node_id].cost:
                    open_nodes[new_node_id].cost = new_node.cost
                    open_nodes[new_node_id].parentID = new_node.parentID
            else:
                open_nodes[new_node_id] = new_node

            heapq.heappush(cost, [open_nodes[new_node_id].cost, open_nodes[new_node_id]])

    return 0, all_nodes


def find_path(end_node):
    """Function to find path to the end node"""
    x_coord = [end_node.x]
    y_coord = [end_node.y]

    node_id = end_node.parentID
    while node_id != -1:
        # current_node = id.parentID
        x_coord.append(node_id.x)
        y_coord.append(node_id.y)
        node_id = node_id.parentID

    x_coord.reverse()
    y_coord.reverse()
    coords = np.vstack((x_coord, y_coord))
    return coords
