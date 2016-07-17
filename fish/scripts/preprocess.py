"""
Run initial processing of raw light sheet data using spark
"""
from glob import glob
from pyspark import SparkConf, SparkContext
from fish.util import fileio
import thunder as td
from skimage.io import imsave
from numpy import save, load

conf = SparkConf().setAppName('preprocessing')
sc = SparkContext(conf=conf)

to_process = glob('/nobackup/ahrens/davis/data/raw/20160614/*/')
do = dict()

do['local_corr'] = False
do['raw_mean'] = False
do['estimate_motion'] = True

dest_fmt = 'klb'
glob_key = 'TM*.{0}'.format(dest_fmt)

def klb_loader(v):
    from pyklb import readfull
    return readfull(v)

def source_conversion(raw_dir):
    raw_fnames = glob(raw_dir + 'TM*.stack')
    raw_fnames.sort()
    if len(raw_fnames) > 0:
        raw_fnames_rdd = sc.parallelize(raw_fnames, numSlices=512)
        raw_fnames_rdd.foreach(lambda v: fileio.image_conversion(v, dest_fmt='klb', wipe=True))
    else:
        print('No stack files found. Doing nothing.')

def prepare_directories(raw_dir):
    from os import makedirs
    from os.path import exists, split
    proc_dir = fileio.dirStructure(raw_dir)
    exp_name = split(split(proc_dir)[0])[1]
    reg_dir = proc_dir + 'reg/'

    if not exists(proc_dir):
        makedirs(proc_dir)

    if not exists(reg_dir):
        makedirs(reg_dir)

    return exp_name, proc_dir, reg_dir


def mean_by_plane(images_object, thr=105):
    from numpy import array

    def thr_mean(vol, thr):
        return array([v[v > thr].mean() for v in vol])
    
    mean_fun = lambda x: array([thr_mean(x[: , :(x.shape[1] // 2), :], thr=thr), thr_mean(x[:, (x.shape[1] // 2):, :], thr=thr)])
    return images_object.map(mean_fun).toarray()


def estimate_motion(fnames, save_reference=True):
    from numpy import arange
    dat = td.images.fromlist(fnames, accessor=klb_loader, engine=sc, npartitions=len(fnames))
    num_frames = dat.shape[0]
    ref_length = 5
    refR = (num_frames // 2) + arange(-ref_length // 2, ref_length // 2)
    ref = td.images.fromlist(fnames[slice(refR[0], refR[-1])], accessor=klb_loader).mean().toarray()
    ref_bc = sc.broadcast(ref)

    # apply some cleaning to images to try to mitigate activity-based translation artifacts
    dat_reg = dat.map(lambda v: v.clip(max=120))

    def proj_reg_batch(fixed, moving):
        from numpy import array, max
        from fish.image.alignment import proj_reg

        tx = proj_reg(fixed, moving, max)
        dxdydz = array(tx.GetParameters())
        return dxdydz

    result = dat_reg.map(lambda v: proj_reg_batch(ref_bc.value, v)).toarray().T
    return result


def transform_images(ims, reg_params):
    reg_bc = sc.broadcast(reg_params)
    dims = ims.shape[1:]
    nrecords = ims.shape[0]
    def warp_image(kv):
        from SimpleITK import AffineTransform    
        from fish.image.alignment import apply_transform_itk
        tx = AffineTransform(len(kv[1].shape))
        tx.SetTranslation(reg_bc.value[:,kv[0][0]])
        return (kv[0], apply_transform_itk(tx, kv[1]).astype('float32'))

    return td.images.fromrdd(ims.tordd().map(warp_image), nrecords=nrecords, dims=dims, ordered=True, dtype='float32')


for raw_dir in to_process:
    print('Begin processing of {0}'.format(raw_dir))
    print('Begin source conversion')
    source_conversion(raw_dir)
    print('Finished with source conversion')

    fnames = glob(raw_dir + glob_key)
    fnames.sort()
    print('Preparing directories')
    exp_name, proc_dir, reg_dir = prepare_directories(raw_dir)
    print('Done preparing directories')

    reg_params_path = reg_dir + 'translation_params.npy'
    raw_mean_path = proc_dir + 'raw_mean.npy'
    local_corr_path = proc_dir + 'local_corr.tif'

    ims = td.images.fromlist(fnames, accessor=klb_loader, engine=sc, npartitions=len(fnames))
    dims = ims.shape[1:]

    if do['raw_mean']:
        print('Begin calculating raw mean')
        raw_mean = mean_by_plane(ims)
        save(raw_mean_path, raw_mean)
        print('Finished calculating raw mean')
    else:
        print('Raw mean already calculated.')
    
    if do ['estimate_motion']:
        print('Begin estimating motion')
        reg_params = estimate_motion(fnames)
        save(reg_params_path, reg_params)
        print('Finished estimating motion')
    else:
        print('Registration params already calculated.')
        reg_params = load(reg_params_path)

    if do['local_corr']:
        print('Begin transforming images')
        ims_tx = transform_images(ims, reg_params)
        print('Done transforming images.')
        print('Begin local correlation')
        #for z in range(dims[0]):
        local_corr = ims_tx[:,0].localcorr().toarray()
        
        imsave(local_corr_path, local_corr, compress=1)
        print('Finished calculating local correlation')
    else:
        print('Local correlation already calculated')

