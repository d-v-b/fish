#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  A simple class for representing a polygonal region of interest.
#
# Davis Bennett
# davis.v.bennett@gmail.com
#
# License: MIT
#


class ROI(object):
    """class for representing a single polygonal ROI"""

    def __init__(self, image=[], x=[], y=[]):
        if image is not None:
            self.image = image
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y

    def __repr__(self):
        return "An ROI containing {0} points".format(len(self.x))

    def reset(self):
        self.x = []
        self.y = []

    def get_mask(self):
        from matplotlib.path import Path
        from numpy import meshgrid, zeros, array, where

        data = self.image
        mask = zeros(data.shape, dtype="uint8")

        if (len(self.y) > 2) & (len(self.x) > 2):

            points = list(zip(self.y, self.x))

            grid = meshgrid(range(data.shape[0]), range(data.shape[1]))
            coords = list(zip(grid[0].ravel(), grid[1].ravel()))

            path = Path(points)
            in_points = array(coords)[where(path.contains_points(coords))[0]]
            in_points = [tuple(x) for x in in_points]
            for t in in_points:
                mask[t] = 255
        else:
            print("Mask requires 3 or more points")

        return mask
