from fish.image.vol import montage_projection
import skimage.io as skio
import os
from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName('max_projection_montage')
sc = SparkContext(conf=conf)
kill_command = '/groups/ahrens/home/bennettd/spark-janelia/spark-janelia destroy'

im_dirs = ['/tier2/ahrens/davis/data/raw/20150110/4DPF_misha_phototaxOMR_onlyCh1_1_20150110_172040/',
           '/tier2/ahrens/davis/data/raw/20150110/4DPF_GFAP_GC6F_OMR_1_20150110_180116/',
           '/tier2/ahrens/davis/data/raw/20150111/5DPF_misha_phototaxOMR_2_20150111_224025/',
           '/tier2/ahrens/davis/data/raw/20150123/gfap_gc6f_7dpf_ori_1_20150123_164315/',
           '/tier2/ahrens/davis/data/raw/20150123/gfap_gc6f_7dpf_omr_cl_1_20150123_195740/']

def make_montage(im_dir):
    exp_name = im_dir.split('/')[-2]
    out_path = '/tier2/ahrens/davis/scratch/' + exp_name + '_x-proj-montage.tif'
    print('Experiment name: {0}'.format(exp_name))
    print('Saving to {0}'.format(out_path))

    montage = montage_projection(im_dir, context=sc)
    skio.imsave(out_path, montage)
    print('Done saving {0} to disk'.format(out_path))
try:
    for im_dir in im_dirs:
        make_montage(im_dir)
except:
    # tear down spark cluster
    os.system(kill_command)

os.system(kill_command)
