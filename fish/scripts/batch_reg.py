import volTools as volt
import numpy as np
import fileTools as ftools
import alignment as align
import os
import thunder as td
from glob import glob
from os.path import split
from pyspark import SparkConf, SparkContext
import pyklb

conf = SparkConf().setAppName('test_spark_batchmode')
sc = SparkContext(conf=conf)
kill_command = '/groups/ahrens/home/bennettd/spark-janelia/spark-janelia destroy'

im_dirs = ['/nobackup/ahrens/davis/data/raw/20160608/6dpf_cy171xcy221_f1_omr_1_20160608_170933/',
           '/nobackup/ahrens/davis/data/raw/20160608/6dpf_cy171xcy221_f2_omr_1_20160608_190404/',
           '/nobackup/ahrens/davis/data/raw/20160608/6dpf_cy171xcy221_f2_omr_1_20160608_190404/', 
           '/nobackup/ahrens/davis/data/raw/20160614/5dpf_cy171xcy221_f1_caudal_omr_1_20160614_183344/',
           '/nobackup/ahrens/davis/data/raw/20160614/5dpf_cy171xcy221_f1_caudal_omr_2_20160614_185018/']


def get_translation(im_dir, sc):
    raw_dir = im_dir
    reg_dir = ftools.dirStructure(raw_dir) + 'reg/'
    exp_name = split(split(ftools.dirStructure(raw_dir))[0])[1]

    print('   raw data: {0}'.format(raw_dir))
    print(' reg params: {0}'.format(reg_dir))
    print('Exp Name: {0}'.format(exp_name))

    if not os.path.exists(reg_dir):
        os.makedirs(reg_dir)
    
    fnames = glob(raw_dir + 'TM*.klb')
    fnames.sort()

    print('Loading {0} images...'.format(len(fnames)))
    klb_loader = lambda v: pyklb.readfull(v)
    dat = td.images.fromlist(fnames, accessor = klb_loader, engine=sc, npartitions=len(fnames))
    print('Done loading images')
    num_frames = dat.shape[0]
    dims = dat.first().shape
    # length of reference, in frames
    ref_length = 5

    # reference timerange
    refR = (num_frames // 2) + np.arange(-ref_length // 2, ref_length // 2)

    print('dims = ' + str(dims))
    print('Num frames = {0}'.format(len(fnames)))
    print('Taking reference from frames ' + str(refR[0]) + ' to ' + str(refR[-1]))
    
    ref = td.images.fromtif(raw_dir + 'TM*.klb', start = refR[0], stop = refR[-1]).mean().toarray()
    
    def proj_reg_batch(fixed, moving):
        from numpy import array, max
        from alignment import proj_reg
        tx = proj_reg(fixed, moving, max)
        dxdydz = array(tx.GetParameters())
        return dxdydz
    
    ref_bc = sc.broadcast(ref)
    result = dat.map(lambda v: proj_reg_batch(ref_bc.value, v)).toarray().T
    np.save(reg_dir + 'translation_params.npy', result)

for im_dir in im_dirs:
    get_translation(im_dir, sc)

os.system(kill_command)
