"""
Run initial processing of raw light sheet data using spark
"""
from glob import glob
from pyspark import SparkConf, SparkContext
conf = SparkConf().setAppName('preprocessing')
sc = SparkContext(conf=conf)

to_process = ['/nobackup/ahrens/davis/data/raw/20160608/6dpf_cy171xcy221_f1_omr_1_20160608_170933/']

dest_fmt = 'klb'
glob_key = 'TM*.{0}'.format(dest_fmt)

def prepare_directories(raw_dir):
    pass

def source_conversion(fnames):
    pass

def raw_mean(fnames, target):
    pass

def estimate_motion(fnames, target):
    pass

for raw_dir in to_process:
    source_conversion(raw_dir)    
    fnames = glob(raw_dir + glob_key)
    fnames.sort()
    raw_mean(fnames)
    estimate_motion(fnames)
    local_corr(fnames)

# If necessary, convert the format / compress the raw data
# If necessary, create a folder(s) for processed data
# Compute the average (ignoring background) of each plane, save to disk as proc/raw_mean.npy   
# Estimate motion, save params to disk as /reg/translation_params.npy
# Compute local correlation after applying registration, save to disk as locorr.tif

