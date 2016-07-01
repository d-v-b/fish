""" file i/o tools for analyzing light sheet data"""


def proj_plot(projs, fig=None, clims='auto', fsize=15, asp=10.0, cmap='gray'):
    """
    projPlot(projs, fig=None, clims='auto', fsize=15,asp=10.0, cmap='gray')

    Plot max projections in 3 subplots.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gspec
    positions = ['bottom', 'top', 'left', 'right']

    # calculate clims if necessary
    if clims == 'auto':
        clims = [np.percentile(p,[0,99.99]) for p in projs]

    ori = 'bottom'

    x = float(projs[1].shape[0])
    y = float(projs[0].shape[0])
    z = float(asp*projs[0].shape[1])

    w = x + z
    h = y + z

    if not fig:
        fig = plt.figure(figsize=(fsize, fsize * h/w))

    # number of subplots in x and y
    plotsY = 2
    plotsX = 2

    h_ratio = [y, z]
    w_ratio = [x, z]

    # prepare grid of axes for plotting
    gs = gspec.GridSpec(plotsY, plotsX, height_ratios=h_ratio, width_ratios=w_ratio)

    imAx = []

    projZY = projs[0]
    projZX = projs[1]
    projXY = projs[2]

    # (z,y) projection
    imAx.append(plt.subplot(gs[-3]))
    plt.imshow(projZY, aspect = 1/asp, origin=ori, cmap=cmap, clim = clims[0])
    imAx[0].yaxis.set_visible(False)
    imAx[0].yaxis.tick_right()
    [imAx[0].spines[x].set_color('w') for x in positions]

    # (z,x) projection
    imAx.append(plt.subplot(gs[-2]))
    plt.imshow(projZX.T, aspect = asp, origin=ori, cmap=cmap, clim = clims[1])
    [imAx[1].spines[x].set_color('w') for x in positions]

    # (x,y) projection
    imAx.append(plt.subplot(gs[-4]))
    plt.imshow(projXY.T, origin=ori, cmap=cmap, clim = clims[2])
    [imAx[2].spines[x].set_color('w') for x in positions]
    imAx[2].xaxis.set_visible(False)

    # extra 4th plot
    imAx.append(plt.subplot(gs[-1]))
    plt.axis('off')

    return imAx


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
    param_files = glob(raw_path + 'Ch*.xml')
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
