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
from ..util.fileio import to_dask
from numpy import array
from pathlib import Path


class ZDS(object):

    def __init__(self, experiment_path, affines=None, single_plane=False):
        """
        initialize a zebrascope data structure with a path to a folder containing raw data and metadata
        """
        self.path = experiment_path
        self.exp_name = Path(self.path).parts[-1]
        self.metadata = get_metadata(self.path + 'ch0.xml')
        self.metadata['volume_rate'] = get_stack_freq(self.path)[0]
        self.files = array(sorted(glob(self.path + 'TM*')))

        if single_plane is False:
            self.shape = (len(self.files), *self.metadata['dimensions'][::-1])
        else:
            self.shape = (len(self.files) * self.metadata['dimensions'][-1], 1, *self.metadata['dimensions'][:-1][::-1])

        try:
            self.data = to_dask(self.files)

            if single_plane:
                self.data = self.data.reshape(self.shape).rechunk((1, *self.shape[1:]))

        except KeyError:
            print('Could not create dask aray from images. Check their format.')
            self.data = None



        self._affines = affines
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

    @reference.setter
    def reference(self, value):
        self._reference = value

    def __repr__(self):
        return 'Experiment name: {0} \nShape: {1}'.format(self.exp_name, self.shape)


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


def get_stack_freq(path):
    """
    Get the temporal data from the Stack_frequency.txt file found in
    path. Return volumetric sampling rate in Hz,
    total recording length in seconds, and total number
    of volumes in a tuple.
    """
    f = open(path + 'Stack_frequency.txt')
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
