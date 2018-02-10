

# define readers and writers for images

def _tif_reader(tif_path):
    from skimage.io import imread
    return imread(tif_path)


def _tif_writer(tif_path, image):
    from skimage.io import imsave
    imsave(tif_path, image)

    
def _stack_reader(stack_path):
    from fish.image import vol as volt
    from numpy import fromfile
    from os.path import sep, split
    dims = volt.get_stack_dims(split(stack_path)[0] + sep)
    im = fromfile(stack_path, dtype='int16')
    im = im.reshape(dims[-1::-1])
    return im


def _stack_writer(stack_path, image):
    raise NotImplementedError

    
def _klb_reader(klb_path):
    from pyklb import readfull
    # pyklb whines if it doesn't get a python string
    return readfull(str(klb_path))


def _klb_writer(klb_path, image):
    from pyklb import writefull
    writefull(image, str(klb_path))

    
def _h5_reader(h5_path):
    from h5py import File
    with File(h5_path, 'r') as f:
        return f['default'].value

    
def _h5_writer(h5_path, data):
    from h5py import File
    from os import remove
    from os.path import exists

    if exists(h5_path):
        remove(h5_path)

    with File(h5_path, 'w') as f:
        f.create_dataset('default', data=data, compression='gzip', chunks=True, shuffle=True)
        f.close()

    
readers = dict()
readers['stack'] = _stack_reader
readers['tif'] = _tif_reader
readers['klb'] = _klb_reader
readers['h5'] = _h5_reader

writers = dict()
writers['stack'] = _stack_writer
writers['tif'] = _tif_writer
writers['klb'] = _klb_writer
writers['h5'] = _h5_writer


def read_image(fname):
    """
    Load .stack, .tif, .klb, or .h5 data and return as a numpy array

    fname : string, path to image file

    """
    # Get the file extension for this file, assuming it is the last continuous string after the last period
    fmt = fname.split('.')[-1]
    return readers[fmt](fname)


def write_image(fname, data):
    """
    Write a numpy array as .stack, .tif, .klb, or .h5 file

    fname : string, path to image file
    
    data : numpy array to be saved to disk
    
    """
    # Get the file extension for this file, assuming it is the last continuous string after the last period
    fmt = fname.split('.')[-1]
    return writers[fmt](fname, data)


def read_images(fnames, parallelism=None):
    """
    Load a sequence of images

    fnames : iterable of file paths

    parallelism : None if no parallelism, an int to indicate the number of processes to use, -1 means use all
    """
    from numpy import array

    # Get the file format of the images
    fmt = fnames[0].split('.')[-1]
    if parallelism is None:
        result = array([readers[fmt](fn) for fn in fnames])

    else:
        if isinstance(parallelism, int):
            from multiprocessing import Pool, cpu_count
            if parallelism == -1:
                num_cores = cpu_count()
            else:
                num_cores = min(parallelism, cpu_count())

            with Pool(num_cores) as pool:
                result = array(pool.map(readers[fmt], fnames))

    return result

#todo: refactor this using the same style as the _writers
def image_conversion(source_path, dest_fmt, wipe=False):
    """
    Convert image from one format to another, optionally erasing the source image

    image_path : string
        Path to image to be converted.
    wipe : bool
        If True, delete the source image after successful conversion

    """

    from numpy import array_equal
    from os import remove
    # the name of the file before format extension
    source_name = source_path.split('.')[0]
    
    dest_path = source_name + '.' + dest_fmt
    source_image = read_image(source_path)
    write_image(dest_path, source_image)

    if wipe:
        check_image = read_image(dest_path)
        if array_equal(check_image, source_image):
            remove(source_path)
        else:
            print('{0} and {1} differ... something went wrong!'.format(source_path, dest_path))
