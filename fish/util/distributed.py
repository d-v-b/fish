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


def get_cluster():
    from dask_drmaa import DRMAACluster
    import os
    # todo: make these arguments function properly
    output_path = '/groups/ahrens/home/bennettd/dask_tmp/'
    error_path = outputPath
    cluster_kwargs_pass = {}
    cluster_kwargs_pass.setdefault(
            'template',
            {
                'args': [
                '--nthreads', '1',
                '--local-directory', '/scratch/' + os.environ['USER']],
                'jobEnvironment': os.environ
            }
        )
    cluster = DRMAACluster(**cluster_kwargs_pass, errorPath=error_path, outputPath=output_path)
    return cluster
