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

from ..image.vol import get_metadata, get_stack_freq, get_stack_dims
from glob import glob
from ..util.fileio import read_image
from numpy import ndarray, squeeze, array
from pathlib import Path


class ZDS(object):

    def __init__(self, experiment_path, parallelism=1):
        """
        initialize a zebrascope data structure with a path to a folder containing raw data and metadata
        """
        # todo: properly handle single-plane recordings
        self.path = experiment_path
        self.exp_name = Path(self.path).parts[-1]
        self.metadata = get_metadata(self.path + 'ch0.xml')
        self.metadata['volume_rate'] = get_stack_freq(self.path)[0]
        self.files = array(sorted(glob(self.path + 'TM*')))
        self.shape = (len(self.files), *get_stack_dims(self.path))
        self.paralellism = parallelism
    
    def __repr__(self):
        return 'Experiment name: {0} \nShape: {1}'.format(self.exp_name, self.shape)
        
    
    def __getitem__(self, item):
        #todo: raise an error when we try to index out of bounds
        # coerce input to slice using code from bolt
        if isinstance(item, int):
            item = tuple([slicify(item, self.shape[0])])
        if isinstance(item, tuple):
            item = tuple([slicify(i, n) if isinstance(i, int) else i for i, n in zip(item, self.shape[:len(item)])])
        if isinstance(item, (list, ndarray)):
            item = (item,)
        if isinstance(item, slice):
            item = (item, *(slice(None),) * len(self.shape[1:]))
        # figure out which files we will be working with
        fnames = self.files[item[0]]

        if len(fnames) == 1:
            result = read_image(fnames, roi=item[1:])
        else:
            result = read_image(fnames, roi=item[1:], parallelism=self.paralellism)

        return squeeze(result)


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
