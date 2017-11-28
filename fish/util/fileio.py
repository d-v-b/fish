

# define loaders for images
def _tif_loader(tif_path):
    from skimage.io import imread
    return imread(tif_path)


def _stack_loader(stack_path):
    from fish.image import vol as volt
    from numpy import fromfile
    from os.path import sep, split
    dims = volt.get_stack_dims(split(stack_path)[0] + sep)
    im = fromfile(stack_path, dtype='int16')
    im = im.reshape(dims[-1::-1])
    return im


def _klb_loader(klb_path):
    from pyklb import readfull
    # pyklb whines if it doesn't get a python string
    return readfull(str(klb_path))


def _h5_loader(h5_path):
    from h5py import File
    with File(h5_path, 'r') as f:
        return f['default'].value

loaders = dict()
loaders['stack'] = _stack_loader
loaders['tif'] = _tif_loader
loaders['klb'] = _klb_loader
loaders['h5'] = _h5_loader


def load_image(fname):
    """
    Load .stack, .tif, .klb, or .h5 data and return as a numpy array

    fname : string, path to image file

    """
    # Get the file extension for this file, assuming it is the last continuous string after the last period
    fmt = fname.split('.')[-1]
    return loaders[fmt](fname)


def load_images(fnames, parallelism=None):
    """
    Load a sequence of images

    fnames : iterable of file paths

    parallelism : None if no parallelism, an int to indicate the number of processes to use, -1 means use all
    """
    from numpy import array

    # Get the file format of the images
    fmt = fnames[0].split('.')[-1]
    if parallelism is None:
        result = array([loaders[fmt](fn) for fn in fnames])

    else:
        if isinstance(parallelism, int):
            from multiprocessing import Pool, cpu_count
            if parallelism == -1:
                num_cores = cpu_count()
            else:
                num_cores = min(parallelism, cpu_count())

            with Pool(num_cores) as pool:
                result = array(pool.map(loaders[fmt], fnames))

    return result

#todo: refactor this using the same style as the _writers
def image_conversion(source_path, dest_fmt, wipe=False):
    """
    Convert uint16 image from .stack or .tif format to .klb/hdf5 format, optionally erasing the source image

    image_path : string
        Path to image to be converted.
    wipe : bool
        If True, delete the source image after successful conversion

    """

    from numpy import array_equal
    from os import remove

    source_name = source_path.split('.')[0]
    dest_path = source_name + '.' + dest_fmt


    def klb_writer(data, klb_path):
        from pyklb import writefull
        writefull(data, klb_path)

    def h5_writer(data, h5_path):
        from h5py import File
        from os.path import exists

        if exists(h5_path):
            remove(h5_path)

        f = File(h5_path, 'w')
        f.create_dataset('default', data=data, compression='gzip', chunks=True, shuffle=True)
        f.close()

    if dest_fmt == 'klb':
        dest_writer = klb_writer

    elif dest_fmt == 'h5':
        dest_writer = h5_writer

    source_image = load_image(source_path)
    dest_writer(source_image, dest_path)

    if wipe:
        check_image = load_image(dest_path)
        if array_equal(check_image, source_image):
            remove(source_path)
        else:
            print('{0} and {1} differ... something went wrong!'.format(source_path, dest_path))
