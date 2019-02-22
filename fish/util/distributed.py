#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Boilerplate for setting up a dask.distributed environment on the janelia compute cluster.
#
# Davis Bennett
# davis.v.bennett@gmail.com
#
# License: MIT
#


def get_jobqueue_cluster(walltime='12:00', cores=1, local_directory=None, memory='16GB', **kwargs):
    """
    Instantiate a dask_jobqueue cluster using the LSF scheduler on the Janelia Research Campus compute cluster.
    This function wraps the class dask_jobqueue.LSFCLuster and instantiates this class with some sensible defaults.
    Extra kwargs added to this function will be passed to LSFCluster().
    The full API for the LSFCluster object can be found here:
    https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.LSFCluster.html#dask_jobqueue.LSFCluster

    """
    from dask_jobqueue import LSFCluster
    import os

    if local_directory is None:
        local_directory = '/scratch/' + os.environ['USER'] + '/'

    cluster = LSFCluster(queue='normal',
                         walltime=walltime,
                         cores=cores,
                         local_directory=local_directory,
                         memory=memory,
                         **kwargs)
    return cluster


def get_drmaa_cluster():
    """
    Instatiate a DRMAACluster for use with the LSF scheduler on the Janelia Research Campus compute cluster. This is a
    wrapper for dask_drmaa.DRMMACluster that uses reasonable default settings for the dask workers. Specifically, this
    ensures that dask workers use the /scratch/$USER directory for temporary files and also that each worker runs on a
    single core. This wrapper also directs the $WORKER.err and $WORKER.log files to /scratch/$USER.

    """
    from dask_drmaa import DRMAACluster
    import os
    
    # we need these on each worker to prevent multithreaded numerical operations
    pre_exec = ('export NUM_MKL_THREADS=1',
                'export OPENBLAS_NUM_THREADS=1',
                'export OPENMP_NUM_THREADS=1',
                'export OMP_NUM_THREADS=1')
    local_directory = '/scratch/' + os.environ['USER']
    output_path = ':' + local_directory
    error_path = output_path
    cluster_kwargs_pass = {}
    cluster_kwargs_pass.setdefault(
        'template',
        {'args': ['--nthreads', '1', '--local-directory', local_directory],
         'jobEnvironment': os.environ,
         'outputPath': output_path,
         'errorPath': error_path}
    )
    cluster_kwargs_pass['preexec_commands'] = pre_exec
    cluster = DRMAACluster(**cluster_kwargs_pass)
    return cluster


def get_downsampled_baseline(data, factor=None, keyframes=None, axis=0, perc=None, window=None, mode='reflect'):
    """
    Generate a dask array that will take the non-sliding windowed percentile of input data along the first axis.
    Note that there is no handling of edges. Values of keyframes at the edges of the range will be result in
    the percentile being estimated over fewer values.

    Returns the keyframes and a stacked dask array where each element is the lazy percentile estimated over the
    0th...len(keyframes)th window. The dtype of this array will be float32.

    keyframes : numpy array, timepoints at which to take the windowed percentile.

    data : dask array.

    factor: integer, the downsampling factor to apply to the data.

    keyframes: sequence of integers, only used if `factor` is not supplied. These are the timepoints at which the
               downsampled baseline will be evaluated.

    axis : integer, axis along which to take the percentile. defaults to 0.

    perc : integer between 0 and 100. This is the percentile that will be measured in each window.

    window : integer, size of the window to use for computing the percentile.

    mode : string, specifies how values at the boundary should be handled. Only supported mode is 'reflect'

    """

    from numpy import linspace, arange, percentile
    from dask.array import stack

    if factor is not None:
        keyframes = linspace(0, data.shape[axis]-1, factor, dtype='int')
    elif keyframes is None:
        raise ValueError('Either factor or keyframes must be specified.')

    window_inds = arange(-(window // 2), 1 + (window // 2))
    inds = window_inds + keyframes.reshape(-1, 1)

    def get_perc(v):
        return percentile(v, perc, axis=axis).astype('float32')

    rechunked = []

    for i in inds:
        new_chunks = ['auto'] * data.ndim
        new_chunks[axis] = len(i)
        new_chunks = tuple(new_chunks)
        # lower edge
        if i[0] < 0:
            if mode == 'reflect':
                invalid = i[i < 0]
                i[i < 0] = arange(len(invalid))

        # upper edge
        if i[-1] >= data.shape[axis]:
            if mode == 'reflect':
                invalid = i[i >= data.shape[axis]]
                i[i >= data.shape[axis]] = arange(data.shape[axis]-1, (data.shape[axis] - 1) - len(invalid), -1)

        res = data[i].rechunk(new_chunks)
        rechunked.append(res)

    stacked = stack([r.map_blocks(get_perc, dtype='float32', drop_axis=axis) for r in rechunked])

    return keyframes, stacked
