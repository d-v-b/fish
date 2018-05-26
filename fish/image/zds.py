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
        self.shape = (len(self.files), *self.metadata['dimensions'][::-1])
        self.paralellism = parallelism
        self._affines = None
        self._reference = None

    @property
    def affines(self):
        return self._affines

    @affines.setter
    def affines(self, value):
        if value.shape[0] != len(self.files):
            raise ValueError('Length of affines must match length of the first axis of the data.')
        self._affines = value

    @property
    def reference(self):
        return self._reference

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


def get_metadata(param_file):
    """
    Parse imaging metadata file, returning a dictionary of imaging parameters

    param_file : str, .xml file containing metadata
    """

    import xml.etree.ElementTree as ET
    from lxml import etree
    from numpy import array

    parser = etree.XMLParser(recover=True)

    exp_dict = {}
    root = ET.parse(param_file, parser=parser).getroot()

    for r in root.findall('info'):
        exp_dict[r.keys()[0]] = r.items()[0][1]

    # convert dimensions from a string formatted 'X_sizexY_sizexZsize' to a numpy array
    if type(exp_dict['dimensions']) is str:
        exp_dict['dimensions'] = array(exp_dict['dimensions'].split('x')).astype('int')

    # convert z step from string to float
    if type(exp_dict['z_step']) is str:
        exp_dict['z_step'] = float(exp_dict['z_step'])

    return exp_dict


def get_stack_freq(inDir):
    """
    Get the temporal data from the Stack_frequency.txt file found in
    directory inDir. Return volumetric sampling rate in Hz,
    total recording length in S, and total number
    of planes in a tuple.
    """
    f = open(inDir + 'Stack_frequency.txt')
    times = [float(line) for line in f]

    # third value should be an integer
    times[2] = int(times[2])

    return times


def rearrange_bidirectional_stack(stack_data):
    """
    Re-arrange the z planes in data that were acquired bidirectionally. Convert from temporal order to spatial order.
    For stacks with an even number of planes, the odd-numbered planes are acquired first, and vice versa.
    For example, a stack with 8 total planes with be acquired in this order: 1, 3, 5. 7, 6, 4, 2, 0

    stack_data: 3-dimensional numpy array

    returns a 3-dimensional numpy array with the same values as stack_data but re-arranged.
    """

    from numpy import zeros, ceil
    z = stack_data.shape[0]
    midpoint = int(ceil(z / 2))
    z_range_old = range(z)
    z_range_new = zeros(z, dtype='int')

    if (z % 2) == 0:
        z_range_new[1::2] = z_range_old[:midpoint]
        z_range_new[0::2] = z_range_old[midpoint:][::-1]
        return stack_data[z_range_new]
    else:
        z_range_new[0::2] = z_range_old[:midpoint]
        z_range_new[1::2] = z_range_old[midpoint:][::-1]
        return stack_data[z_range_new]
