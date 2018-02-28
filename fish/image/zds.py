#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  A class for wrapping raw data + metadata from the Ahrens Lab light sheet microscope
#
# Davis Bennett
# davis.v.bennett@gmail.com
#
# License: MIT
#

from ..image.vol import get_metadata, get_stack_freq
from glob import glob
from ..util.fileio import read_image
from numpy import ndarray


class ZDS(object):

    def __init__(self, experiment_path):
        """
        initialize a zebrascope data structure with a path to a folder containing raw data and metadata
        """
        self.path = experiment_path
        self.metadata = get_metadata(self.path + 'Ch0.xml')
        self.timing = get_stack_freq(self.path)
        self.files = sorted(glob(self.path + 'TM*'))
        self.shape = (len(self.files), *read_image(self.files[0]).shape)

    def __getitem__(self, item):

        # raise NotImplementedError

        if isinstance(item, int):
            item = tuple([slicify(item, self.shape[0])])
        if isinstance(item, tuple):
            item = tuple([slicify(i, n) if isinstance(i, int) else i for i, n in zip(item, self.shape[:len(item)])])
        if isinstance(item, (list, ndarray)):
            item = (item,)

        # figure out which files we will be working with
        fnames = self.files[item[0]]

        if len(fnames) == 1:
            return read_image()


def slicify(slc, dim):
    """
    Force a slice to have defined start, stop, and step from a known dim.
    Start and stop will always be positive. Step may be negative.
    There is an exception where a negative step overflows the stop needs to have
    the default value set to -1. This is the only case of a negative start/stop
    value.

    This is copied from bolt/bolt/utils.py

    Parameters
    ----------
    slc : slice or int
        The slice to modify, or int to convert to a slice
    dim : tuple
        Bound for slice
    """
    if isinstance(slc, slice):

        # default limits
        start = 0 if slc.start is None else slc.start
        stop = dim if slc.stop is None else slc.stop
        step = 1 if slc.step is None else slc.step
        # account for negative indices
        if start < 0: start += dim
        if stop < 0: stop += dim
        # account for over-flowing the bounds
        if step > 0:
            if start < 0: start = 0
            if stop > dim: stop = dim
        else:
            if stop < 0: stop = -1
            if start > dim: start = dim-1

        return slice(start, stop, step)

    elif isinstance(slc, int):
        if slc < 0:
            slc += dim
        return slice(slc, slc+1, 1)

    else:
        raise ValueError("Type for slice %s not recongized" % type(slc))
