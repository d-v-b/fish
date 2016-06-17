"""
Run initial processing of raw light sheet data using spark
"""
from glob import glob
from pyspark import SparkConf, SparkContext
from fish.image import alignment
from fish.util import fileio
import thunder as td

conf = SparkConf().setAppName('preprocessing')
sc = SparkContext(conf=conf)

to_process = ['/nobackup/ahrens/davis/data/raw/20160608/6dpf_cy171xcy221_f1_omr_1_20160608_170933/']

dest_fmt = 'klb'
glob_key = 'TM*.{0}'.format(dest_fmt)
klb_loader = lambda v: pyklb.readfull(v)

def source_conversion(raw_dir):
    raw_fnames = glob(raw_dir + 'TM*.stack')
    raw_fnames.sort()
    if len(raw_fnames) > 0:
        raw_fnames_rdd = sc.parallelize(raw_fnames, numSlices=512)
        raw_fnames_rdd.foreach(lambda v: fileio.image_conversion(v, wipe=True)


def prepare_directories(raw_dir):
    from os import makedirs
    from os.path import exists, split
    proc_dir = fileio.dirStructure(raw_dir)
    exp_name = split(split(ftools.dirStructure(raw_dir))[0])[1]
    reg_dir = proc_dir + 'reg/'

    if not exists(proc_dir):
        makedirs(proc_dir)
    if not exists(reg_dir):
        makedirs(reg_dir)
    return exp_name, proc_dir, reg_dir


def mean_by_plane(images_object, thr=105):
    def thr_mean(vol, thr):
        from numpy import array
        return array([v[v > thr].mean() for v in vol])
    
    mean_fun = lambda x: np.array([thr_mean(x[: ,:(x.shape[1] // 2), :], thr=thr), thr_mean(x[:, (x.shape[1] // 2):, :], thr=thr)])
    return images_object.map(mean_fun).toarray()


def estimate_motion(fnames, target_dir):    
    dat = td.images.fromlist(fnames, accessor = klb_loader, engine=sc, npartitions=len(fnames))
    num_frames = dat.shape[0]
    ref_length = 5
    refR = (num_frames // 2) + np.arange(-ref_length // 2, ref_length // 2)
    ref = td.images.fromlist(fnames[slice(refR[0], refR[-1])], accessor=klb_loader).mean().toarray()
    ref_bc = sc.broadcast(ref)

    def proj_reg_batch(fixed, moving):
        from numpy import array, max
        from alignment import proj_reg

        tx = proj_reg(fixed, moving, max)
        dxdydz = array(tx.GetParameters())
        return dxdydz

    result = dat.map(lambda v: proj_reg_batch(ref_bc.value, v)).toarray().T
    return result


def apply_transform(ims, transform):
    pass


def local_corr(images):
    return images.localcorr().toarray()


for raw_dir in to_process:
    source_conversion(raw_dir)    
    fnames = glob(raw_dir + glob_key)
    fnames.sort()
    exp_name, proc_dir, reg_dir = prepare_directories(raw_dir)
    reg_params_path = reg_dir + 'translation_params.npy'
    raw_mean_path = proc_dir + 'raw_mean.npy'
    ims = td.images.fromlist(fnames, accessor=klb_loader, engine=sc, npartitions=len(fnames)) 
    raw_mean = mean_by_plane(ims)
    np.save(raw_mean_path, raw_mean)
    reg_params = estimate_motion(fnames)
    np.save(reg_params_path, reg_params)
    ims_tx = apply_transform(ims, reg_params)

