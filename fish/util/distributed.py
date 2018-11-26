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

