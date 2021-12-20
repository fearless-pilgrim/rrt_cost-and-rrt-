# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import sys
sys.path.append('/home/lift/wzw/segement/planning/rrt-algorithms')

import numpy as np
from src.rrt.rrt_star import RRTStar
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot

X_dimensions = np.array([(0, 100), (0, 100)])  # dimensions of Search Space
# obstacles
Obstacles = []
obstacle = []
for i in range(1000):
    obstacle = np.random.randint(X_dimensions[:,0],X_dimensions[:,1])
    print(tuple(obstacle))
    Obstacles.append(tuple(obstacle))

print(len(Obstacles))
# for i in range(X_dimensions[:,0]):
#     for j in range(X_dimensions[:,1]):
#         if image[i,j,1] == 255:
#             obstacle = np.append(i,j)
#             Obstacles.append(tuple(obstacle))
# Obstacles = np.array([(20, 20, 40, 40), (20, 60, 40, 80), (60, 20, 80, 40), (60, 60, 80, 80)])
x_init = (0, 0)  # starting location
x_goal = (100, 100)  # goal location

Q = np.array([(8, 4)])  # length of tree edges
r = 5  # length of smallest edge to check for intersection with obstacles
max_samples = 2024  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 1  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions, Obstacles)

# create rrt_search
rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
path = rrt.rrt_star()
# print(path)
# plot
plot = Plot("rrt_star_2d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles_self(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)
