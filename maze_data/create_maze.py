""" df_maze.py
    Create a maze using the depth-first algorithm described at
    https://scipython.com/blog/making-a-maze/
    Christian Hill, April 2017.
    Modified July 2020 for DeepThinking.
"""
import random

import numpy as np

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class Cell:
    """A cell in the maze.
    A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.
    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        """Initialize the cell at (x, y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False


class Maze:
    """A Maze, represented as a grid of cells."""

    def __init__(self, nx, ny, ix=0, iy=0):
        """Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).
        """

        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def cell_at(self, x, y):
        """Return the Cell object at (x, y)."""

        return self.maze_map[x][y]

    def __str__(self):
        """Return a (crude) string representation of the maze."""

        nx = self.nx
        ny = self.ny
        maze_rows = ['-' * nx*2]
        for y in range(ny):
            maze_row = ['|']
            for x in range(nx):
                if self.maze_map[x][y].walls['E']:
                    maze_row.append(' |')
                else:
                    maze_row.append('  ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(nx):
                if self.maze_map[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return '\n'.join(maze_rows)

    def write_np(self):
        """Write an SVG image of the maze to filename."""
        beta = 2
        my_numpy_array = np.ones((beta*(self.ny) + 1, beta*(self.nx) + 1))
        # Draw the "South" and "East" walls of each cell, if present (these
        # are the "North" and "West" walls of a neighbouring cell in
        # general, of course).
        for x in range(self.nx):
            for y in range(self.ny):
                if self.cell_at(x, y).walls['S']:
                    my_numpy_array[beta*(y + 1), beta*(x): beta*(x + 1)] = 0
                if self.cell_at(x, y).walls['E']:
                    my_numpy_array[beta*(y) : beta*(y + 1), beta*(x + 1)] = 0
                if self.cell_at(x, y).walls['S'] and self.cell_at(x, y).walls['E']:
                    my_numpy_array[beta*(y + 1), beta*(x + 1)] = 0
        # Draw the North and West maze border, which won't have been drawn
        # by the procedure above.
        my_numpy_array[0, :] = 0
        my_numpy_array[:, 0] = 0
        return np.dstack([my_numpy_array, my_numpy_array, my_numpy_array])

    def find_valid_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self):
        """make a single maze"""
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)

            if not neighbours:
                # We've reached a dead end: backtrack.
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, next_cell = random.choice(neighbours)
            current_cell.knock_down_wall(next_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1


def create_maze(n=20, ix=0, iy=0):
    """Returns numpy array of maze with n columns/rows and a starting point of (0,0)."""
    maze = Maze(n, n, ix, iy)
    maze.make_maze()
    arr = maze.write_np()
    return arr
