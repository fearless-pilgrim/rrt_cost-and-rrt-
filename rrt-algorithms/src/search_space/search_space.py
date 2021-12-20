# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

from flatbuffers.number_types import to_numpy_type
import numpy as np
from rtree import index

from src.utilities.geometry import es_points_along_line,steer,dist_between_points
from src.utilities.obstacle_generation import obstacle_generator


class SearchSpace(object):
    def __init__(self, dimension_lengths, O=None, x_init = None, x_goal = None):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param O: list of obstacles
        """
        # sanity check
        if len(dimension_lengths) < 2:
            raise Exception("Must have at least 2 dimensions")
        self.dimensions = len(dimension_lengths)  # number of dimensions
        # sanity checks
        if any(len(i) != 2 for i in dimension_lengths):
            raise Exception("Dimensions can only have a start and end")
        if any(i[0] >= i[1] for i in dimension_lengths):
            raise Exception("Dimension start must be less than dimension end")
        self.dimension_lengths = dimension_lengths  # length of each dimension
        p = index.Property()
        p.dimension = self.dimensions
        self.obs = O
        if not self.obstacle_free(x_init) or not self.obstacle_free(x_goal):
            raise Exception("Obstacle has incorrect dimension definition")
        # if O is None:
        #     self.obs = index.Index(interleaved=True, properties=p)
        # else:
        #     # r-tree representation of obstacles
        #     # sanity check
        #     if any(len(o) / 2 != len(dimension_lengths) for o in O):
        #         raise Exception("Obstacle has incorrect dimension definition")
        #     if any(o[i] >= o[int(i + len(o) / 2)] for o in O for i in range(int(len(o) / 2))):
        #         raise Exception("Obstacle start must be less than obstacle end")
        #     self.obs = index.Index(obstacle_generator(O), interleaved=True, properties=p)

    def obstacle_free(self, x):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """

        return self.obs.count(x) == 0


    def sample_free(self):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:  # sample until not inside of an obstacle
            x = self.sample()
            if self.obstacle_free(x):
                return x

    def collision_free(self, start, end, r):
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when checking for collisions
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        # points = es_points_along_line(start, end, r)
        # print(points)
        # coll_free = all(map(self.obstacle_free, points))
        # return coll_free
        new = []
        d = dist_between_points(start, end)
        n_points = int(np.ceil(d / r))

        if n_points > 1:
            step = d / (n_points - 1)
            for i in range(n_points):
                next_point = steer(start, end, i * step)
                new = np.append(int(next_point[0]),int(next_point[1]))
                if not self.obstacle_free(tuple(new)):
                    return False
        return True

    def sample(self):
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """
        y = []
        x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        # x = np.random.randint(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        y = np.append((int)(x[0]),(int)(x[1]))
        y = tuple(y)
        return tuple(y)