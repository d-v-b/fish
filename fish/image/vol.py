""" tools for analyzing light sheet data"""


def local_corr(images, offset=[0,1,1]):
    """
    Correlate each image in a distributed set of images with a shifted copy of itself. Returns an rdd of
    correlation coefficients.

    images : thunder.images object
    offset : the shift, in pixels, to apply to each image before correlation

    """
    from scipy.ndimage.interpolation import shift
    
    def correlate_signals(s1, s2):
        from numpy import corrcoef
        return corrcoef(s1, s2)[0][1]
    
    images_shifted = images.map(lambda v: shift(v.astype('float32'), offset, mode='reflect')).astype('float16')
    joined = images.toseries().tordd().join(images_shifted.toseries().tordd())
    
    return joined.mapValues(lambda v: correlate_signals(v[0], v[1]))


def get_metadata(param_file):
    """
    Parse imaging metadata file, returning a dictionary of imaging parameters

    param_file : str, .xml file containing metadata
    """

    import xml.etree.ElementTree as ET
    from numpy import array

    exp_dict = {}
    root = ET.parse(param_file).getroot()

    for r in root.findall('info'):
        exp_dict[r.keys()[0]] = r.items()[0][1]

    # convert dimensions from a string formatted 'X_sizexY_sizexZsize' to a numpy array
    if type(exp_dict['dimensions']) is str:
        exp_dict['dimensions'] = array(exp_dict['dimensions'].split('x')).astype('int')

    # convert z step from string to float
    if type(exp_dict['z_step']) is str:
        exp_dict['z_step'] = float(exp_dict['z_step'])

    return exp_dict


def get_stack_dims(inDir):
    """
    :param inDir: a string representing a path to a directory containing metadata
    :return: dims, a list of integers representing the xyz dimensions of the data
    """
    import xml.etree.ElementTree as ET
    from os.path import split

    channel = 0
    if split(split(inDir)[0])[1] == 'CHN01':
        channel = 1

    dims = ET.parse(inDir + 'ch' + str(channel) + '.xml')
    root = dims.getroot()

    for info in root.findall('info'):
        if info.get('dimensions'):
            dims = info.get('dimensions')

    dims = dims.split('x')
    dims = [int(float(num)) for num in dims]

    return dims


def proj_plot(volume, proj_fun, clims='auto', figsize=15, aspect=10, cmap='gray'):
    """
    Project a volume along 3 axes using a user-supplied function

    volume : data to be projected.
        3D numpy array

    proj_fun : function to apply along each axis.
        Some function that takes axis as an argument, e.g. np.amax()

    clims : clims to use when displaying projections.
        String or iterable with 3 elements. Default is 'auto', which means the 0th and 99.99th percentiles
        will be used as the clims for each projection. If not auto, clims should be set to an iterable of length-2
        iterables, each setting the clim for a projection.

    figsize : size of the figure containing the plots
        Float or int

    aspect : aspect ratio of first axis relative to second and third.
        Float or int. We assume here that the first axis has the lowest
        spatial sampling rate and thus that data plotted against that axis must be interpolated.

    cmap : color map used in plots.
    """

    from numpy import percentile, r_
    from matplotlib.pyplot import subplots

    ori = 'lower'

    projs = [proj_fun(volume, axis=axis) for axis in range(volume.ndim)]

    # calculate clims if necessary
    if clims == 'auto':
        clims = percentile(r_[[p.ravel() for p in projs]], (0, 99.99))

    x, y, z = volume.shape[2], volume.shape[1], volume.shape[0] * aspect

    w = x + z
    h = y + z

    wr = x / w
    hr = y / h

    p_xy = projs[0]
    p_zx = projs[1]
    p_zy = projs[2]

    fig, axs = subplots(nrows=2, ncols=2, figsize=(figsize, figsize * h/w))

    axs[0][0].imshow(p_xy, origin=ori, aspect='auto', cmap=cmap, clim=clims[0])
    axs[1][0].imshow(p_zx, origin=ori, aspect = 'auto', cmap=cmap, clim=clims[1])
    axs[0][1].imshow(p_zy.T, origin=ori, aspect = 'auto', cmap=cmap, clim=clims[2])

    axs[0][0].set_position([0, 1-hr, wr, hr])
    axs[0][1].set_position([wr, 1-hr, 1-wr, hr])
    axs[1][0].set_position([0, 0, wr, 1-hr])
    axs[1][1].set_position([wr, 0, 1-wr, 1-hr])
    [ax.axis('off') for ax in axs.ravel()]

    return axs


def get_stack_freq(inDir):
    """
    Get the temporal data from the Stack_frequency.txt file found in
    directory inDir. Return volumetric sampling rate in Hz,
    total recording length in S, and total number
    of planes in a tuple.
    """
    f = open(inDir + 'Stack_frequency.txt')
    times = [float(line) for line in f]

    # third value should be an integer
    times[2] = int(times[2])

    return times


def get_stack_data(raw_path, frameNo=0):
    """
    :rawPath: string representing a path to a directory containing raw data
    :frameNo: int representing the timepoint of the data desired, default is 0
    """

    from numpy import fromfile
    from string import Template
    from os.path import split
    from glob import glob

    channel = 0
    if split(raw_path)[0][-2:] == '01':
        channel = 1
    param_files = glob(raw_path + 'ch*.xml')
    if len(param_files) == 0:
            print('No .xml files found!')

    dims = get_metadata(param_files[0])['dimensions']
    fName = Template('TM${x}_CM0_CHN${y}.stack')
    nDigits_frame = 5
    nDigits_channel = 2
    tmpFName = fName.substitute(x=str(frameNo).zfill(nDigits_frame), y=str(channel).zfill(nDigits_channel))
    im = fromfile(raw_path + tmpFName, dtype='int16')
    im = im.reshape(dims[-1::-1])
    return im


def vid_embed(fname, mimetype):
    """Load the video in the file `fname`, with given mimetype, and display as HTML5 video.
    Credit: Fernando Perez
    """
    from IPython.display import HTML
    video_encoded = open(fname, "rb").read().encode("base64")
    video_tag = '<video controls alt="test" src="data:video/{0};base64,{1}">'.format(mimetype, video_encoded)
    return HTML(data=video_tag)


def volume_mask(vol):
    """
    :param vol: a 3-dimensional numpy array
    :return: mask, a binary mask with the same shape as vol, and mCoords, a list of (x,y,z) indices representing the
    masked coordinates.
    """
    from numpy import array, where
    from scipy.signal import medfilt2d
    from skimage.filter import threshold_otsu
    from skimage import morphology as morph

    filtVol = array([medfilt2d(x.astype('float32')) for x in vol])

    thr = threshold_otsu(filtVol.ravel())
    mask = filtVol > thr
    strel = morph.selem.disk(3)
    mask = array([morph.binary_closing(x, strel) for x in mask])
    mask = array([morph.binary_opening(x, strel) for x in mask])

    z, y, x = where(mask)
    mCoords = zip(x, y, z)

    return mask, mCoords


def filter_flat(vol, mask):
    """
    Flatten an array and return a list of the elements at positions where the binary mask is True.

    vol : ndarray
    mask : binary ndarray or function. If a function, mask must take an ndarray as an argument and return a
    binary mask.
    """
    vol_flat = vol.ravel()

    # if mask is a function, call it on vol to make the mask
    if hasattr(mask, '__call__'):
        mask_flat = mask(vol).ravel()
    else:
        mask_flat = mask.ravel()

    return vol_flat[mask_flat]


def unfilter_flat(vec, mask):
    """
    Reverse the effect of filter_flat by taking a 1d ndarray and assigning each value to a position in an ndarray

    vec : 1-dimensional ndarray
    mask : binary ndarray
    """

    from numpy import zeros
    """
    Fill a binary mask with the values in vec
    """
    mask_flat = mask.ravel()
    vol = zeros(mask.shape).ravel()
    vol[mask_flat == True] = vec

    return vol.reshape(mask.shape)


def kvp_to_array(dims, data, ind=0, baseline=0):
    """ Convert a list of key-value pairs to a volume.

    :param dims: dimensions of the volume to fill with values
    :param data: list of iterables with 2 values, 2-tuples or 2-item lists.
    :param ind: if the value in each key-value pair is itself an iterable, ind specifies which element to use
    :param baseline: fill value for empty spots in the volume
    :return: vol, an ndarray
    """
    from numpy import zeros, array

    vol = zeros(dims, dtype=data[0][1].dtype) + array(baseline).astype(data[0][1].dtype)

    for k, v in data:
        # check if data contains a single value or an iterable
        if hasattr(v, '__iter__'):
            vol[k] = v[ind]
        else:
            vol[k] = v
    return vol


def sub_proj(im, ax, chop=16):
    """
    Max project a volume in chunks along an axis.

    im : numpy array, data to be projected

    ax : int, axis to project along

    chop : int, number of projections to generate

    """
    from numpy import rollaxis

    extra = im.shape[ax] % chop
    montage_dims = list(im.shape)
    montage_dims[ax] //= chop
    montage_dims.insert(ax, chop)

    slices_crop = [slice(None) for x in im.shape]
    slices_crop[ax] = slice(0, extra + 1, 1)

    # remove trailing data by projecting it down to a single plane
    im[slices_crop] = im[slices_crop].max(ax, keepdims=True)

    slices_keep = [slice(None) for x in im.shape]
    slices_keep[ax] = slice(extra, None)
    im_proj = im[slices_keep].reshape(montage_dims).max(ax + 1)
    # stick the axis of projections in the front
    im_proj = rollaxis(im_proj, ax, 0)

    return im_proj


def montage_projection(im_dir, trange=None, context=None):
    """
    Generate a montage of x projections.

    im_dir : str, path to directory containing [x,y,z] data saved as tif
    
    trange : object which can be used for linear indexing, set of timepoints to use

    context : spark context object for parallelization
    """
    import thunder as td
    from glob import glob
    from skimage.util.montage import montage2d
    from skimage.exposure import rescale_intensity as rescale
    import numpy as np
    from pyklb import readfull

    exp_name = im_dir.split('/')[-2]

    print('Exp name: {0}'.format(exp_name))

    fnames = glob(im_dir + 'TM*.klb')
    fnames.sort()

    def klb_loader(v):
        return pyklb.readfull(v)

    ims = td.images.fromlist(fnames, accessor=klb_loader, engine=context)

    print('Experiment dims: {0}'.format(ims.shape))
    
    if trange is None:
        trange = np.arange(ims.shape[0])
    
    ims_cropped = ims[trange].median_filter([1, 3, 3])
    dims = ims_cropped.dims

    #todo: apply registration if available

    from scipy.ndimage import percentile_filter
    float_dtype = 'float32'
    
    def my_dff(y, perc, window): 
        baseFunc = lambda x: percentile_filter(x.astype(float_dtype), perc, window, mode='reflect')
        b = baseFunc(y)
        return ((y - b) / (b + .1))

    dff_fun = lambda v: my_dff(v, 15, 800) 
    chop = 16

    reshape_fun = lambda v: v.reshape(dims[0], dims[1], chop, dims[2] // chop)
    montage_fun = lambda v: montage2d(v.T).T

    def im_fun(v):
        return montage_fun(reshape_fun(v).max(3))
    
    out_dtype = 'uint16'
    
    montage_ims = ims_cropped.map_as_series(dff_fun, value_size=ims_cropped.shape[0], dtype=float_dtype, chunk_size='35').map(im_fun)
    dff_lim = montage_ims.map(lambda v: [v.max(), v.min()]).toarray()
    rescale_fun = lambda v: rescale(v, in_range=(dff_lim.min(), dff_lim.max()), out_range=out_dtype).astype(out_dtype)

    montage_rescaled = montage_ims.map(rescale_fun).toarray()[:,-1::-1,:]
    return montage_rescaled
