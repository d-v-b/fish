#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Tools for visualizing volumetric data
#
# Davis Bennett
# davis.v.bennett@gmail.com
#
# License: MIT
#


from ..util.roi import ROI


def proj_plot(volume, proj_fun, clims='auto', figsize=4, aspect=(1, 1, 1), cmap='gray', interpolation='Lanczos'):
    """
    Project and plot a volume along 3 axes using a user-supplied function, using separate subplots for each projection

    volume : Numpy array, 3D (grayscale data) or 4D (RGB data).
        data to be projected.

    proj_fun : function to apply along each axis.
        Some function of numpy arrays that takes axis as an argument, e.g. numpy.max()

    clims : clims to use when displaying projections.
        String or iterable with 3 elements. Default is 'auto', which means the 0th and 100th percentiles
        will be used as the clims for each projection. If not auto, clims should be set to an iterable of length-2
        iterables, each setting the clim for a projection. This setting is ignored if input array is 4D.

    figsize : size of the figure containing the plots
        Float or int

    aspect : aspect ratios of each axis
        Iterable of floats or ints.

    cmap : color map used in plots. This setting is ignored if input array is 4D
    """

    from numpy import percentile, hstack, swapaxes
    from matplotlib.pyplot import subplots

    ori = 'lower'

    projs = [proj_fun(volume, axis=axis) for axis in range(3)]

    # calculate clims for grayscale if necessary
    if clims == 'auto':
        clims = percentile(hstack([p.ravel() for p in projs]), (0, 100))
        clims = (clims, clims, clims)

    z, y, x = volume.shape[0] * aspect[0], volume.shape[1] * aspect[1], volume.shape[2] * aspect[2]

    w = x + z
    h = y + z

    wr = x / w
    hr = y / h

    p_xy = projs[0]
    p_zx = projs[1]
    p_zy = swapaxes(projs[2], 0, 1)

    fig, axs = subplots(nrows=2, ncols=2, figsize=(figsize, figsize * h/w))

    axs[0][0].imshow(p_xy, origin=ori, aspect='auto', cmap=cmap, clim=clims[0], interpolation=interpolation)
    axs[1][0].imshow(p_zx, origin=ori, aspect='auto', cmap=cmap, clim=clims[1], interpolation=interpolation)
    axs[0][1].imshow(p_zy, origin=ori, aspect='auto', cmap=cmap, clim=clims[2], interpolation=interpolation)

    axs[0][0].set_position([0, 1-hr, wr, hr])
    axs[0][1].set_position([wr, 1-hr, 1-wr, hr])
    axs[1][0].set_position([0, 0, wr, 1-hr])
    axs[1][1].set_position([wr, 0, 1-wr, 1-hr])
    [ax.axis('off') for ax in axs.ravel()]

    return axs


def proj_fuse(data, fun, aspect=(1, 1, 1), fill_value=0, arrangement=[0,1,2]):
    """
    Project a volume along 3 axes using a user-supplied function, returning a 2D composite of projections. If the input
    array has the shape [z,y,x], the output shape will be: [z * aspect_z + y * aspect + y, z * aspect_z + x * aspect_x]

    data : 3D numpy array
        Volumetric data to be projected.

    fun : Function to apply along each axis of the input array.
        A function of numpy arrays that takes an axis as a second argument, e.g. numpy.max()

    aspect : Iterable of floats or ints.
        Amount to scale each axis when forming the composite of projections

    fill_value : int or float
        Default value in the array that this function returns. The corner of the

    """
    from numpy import array, zeros
    from skimage.transform import resize

    old_dims = array(data.shape)
    new_dims = array(aspect) * old_dims

    stretched = zeros([new_dims[1] + new_dims[0], new_dims[2] + new_dims[0]]) + fill_value
    projs = []

    for axis, dim in enumerate(new_dims):
        indexer = list(range(len(new_dims)))
        indexer.pop(axis)
        projs.append(resize(fun(data, axis), new_dims[indexer], mode='constant', preserve_range=True))
    if arrangement == [0, 1, 2]:
        stretched[:new_dims[1], new_dims[2]:] = projs[2].T
        stretched[new_dims[1]:, :new_dims[2]] = projs[1]
        stretched[:new_dims[1], :new_dims[2]] = projs[0]
    elif arrangement == [2, 0, 1]:
        stretched[:new_dims[1], :new_dims[0]] = projs[2].T[:, ::-1]
        stretched[new_dims[1]:, new_dims[0]:] = projs[1]
        stretched[:new_dims[1], new_dims[0]:] = projs[0]
    else:
        raise ValueError('Arrangement must be [0, 1, 2] or [2, 0, 1]')

    return stretched


def apply_cmap(data, cmap='gray', clim='auto'):
    """
    Apply a matplotlib colormap to a 2D or 3D numpy array and return the rgba data in uint8 format

    data : 2D or 3D numpy array

    cmap : string denoting a matplotlib colormap
        Colormap used for displaying frames from data. Defaults to 'gray'.

    clim : length-2 list, tuple, or ndarray, or string
        Upper and lower intensity limits to display from data. Defaults to 'auto'
        If clim='auto', the min and max of data will be used as the clim.
        Before applying the colormap, data will be clipped from clim[0] to clim[1].
    """

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from numpy import array

    if clim == 'auto':
        clim = data.min(), data.max()

    sm = ScalarMappable(Normalize(*clim, clip=True), cmap)
    rgba = np.array([sm.to_rgba(d, bytes=True) for d in data])

    return rgba


def depth_project(data, axis=0, cmap='jet', clim='auto'):
    """
    Generate an RGB "depth projection" of a 3D numpy array. After the input data are normalized to [0,1], planes along
    the projection axis are mapped to positions in a linear RGB colormap. Thus for each plane there is an RGB color,
    and each point in the plane is multiplied with that RGB color. The output is the sum of the values along the
    projection axis, i.e. a 3D array with the last dimension containing RGB values.

    data : 3D numpy array

    axis : int denoting an axis to project over

    cmap : string denoting a matplotlib colormap

    clim : string, or list or tuple with length 2.
        This argument determines the minimum and maximum intensity values to use before rescaling the data to the range
        [0,1]. The default value, 'auto', specifies that the minimum and maximum values of the input data will be mapped
        to [0,1].
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from numpy import linspace, zeros, array
    from skimage.exposure import rescale_intensity as rescale

    if clim == 'auto':
        clim = data.min(), data.max()
    sm = ScalarMappable(Normalize(0, 1, clip=True), cmap)

    cm = sm.to_rgba(linspace(0, 1, data.shape[axis]))
    cvol = zeros((*data.shape, 4))
    data_r = rescale(data.astype('float32'), in_range=clim, out_range=(0, 1))
    data_r = array([data_r] * cm.shape[-1]).transpose(1, 2, 3, 0)
    for ind in range(cvol.shape[axis]):
        slices = [slice(None)] * cvol.ndim
        slices[axis] = ind
        slices = tuple(slices)
        cvol[slices] = cm[ind] * data_r[slices]
    cvol[:, :, :, -1] = 1
    proj = cvol.sum(axis)

    return proj


def nparray_to_video(fname, data, clim='auto', cmap='gray', codec='h264', fps=24,
                     ffmpeg_params=['-pix_fmt', 'yuv420p']):
    """
    Save 3D (t, y, x) numpy array to disk as movie. Uses matplotlib colormaps for rescaling / coloring data,
    and uses moviepy.editor.ImageSequenceClip for movie creation.

    Warning : this function duplicates the input data in memory.

    fname : string
        Filename with extension (.avi, .mp4, etc).

    data : 3D numpy array
        Each 2D array along the first axis of data will be a frame in the movie.

    clim : length-2 list, tuple, or ndarray, or string
        Upper and lower intensity limits to display from data. Defaults to 'auto'
        If clim='auto', the min and max of data will be used as the clim.
        Before applying the colormap, data will be clipped from clim[0] to clim[1].

    cmap : string denoting a matplotlib colormap
        Colormap used for displaying frames from data. Defaults to 'gray'.

    codec :  string
        Which video codec to use. Defaults to 'h264'. See moviepy.editor.ImageSequenceClip.writevideofile.

    fps : int or float
        Frames per second of the movie. Defaults to 24.

    ffmpeg_params : list of strings
        Arguments sent to ffmpeg during movie creation. Defaults to ['-pix_fmt', 'yuv420p'], which is necessary for
        creating movies that OSX understands.


    """
    from moviepy.editor import ImageSequenceClip

    dur = data.shape[0] / fps

    # ffmpeg errors if the dimensions of each frame are not divisible by 2
    if data.shape[1] % 2 == 1:
        data = np.pad(data, ((0, 0), (0, 1), (0, 0)), mode='minimum')
    elif data.shape[2] % 2 == 1:
        data = np.pad(data, ((0, 0), (0, 0), (0, 1)), mode='minimum')

    data_rgba = apply_cmap(data, cmap=cmap, clim=clim)
    clip = ImageSequenceClip([d for d in data_rgba], fps=fps)
    clip.write_videofile(fname, audio=False, codec=codec, fps=fps, ffmpeg_params=ffmpeg_params)


class RoiDrawing(object):
    """Class for drawing ROI on matplotlib figures"""

    def __init__(self, ax, image_data):
        self.image_axes = ax
        self._focus_index = -1
        self.image_data = image_data
        self.lines = []
        self.rois = []
        self.cid_press = self.image_axes.figure.canvas.mpl_connect('button_press_event', self.onpress)
        self.cid_release = self.image_axes.figure.canvas.mpl_connect('button_press_event', self.onpress)
        self.masks = []
        self.selector = []

    @property
    def focus_index(self):
        return self._focus_index

    @focus_index.setter
    def focus_index(self, value):

        if value < 0:
            value = 0
        if value > (len(self.rois) - 1):
            self.new_roi()
        self._focus_index = value

    def focus_incr(self, event=None):
        self.focus_index += 1

    def focus_decr(self, event=None):
        self.focus_index -= 1

    def new_roi(self, event=None):
        self.lines.append(self.image_axes.plot([0], [0])[0])
        self.rois.append(ROI(image=self.image_data, x=[], y=[]))
        self.masks.append(None)

    def onpress(self, event):
        from matplotlib.widgets import Lasso

        if event.inaxes != self.image_axes:
            return
        if self.image_axes.figure.canvas.widgetlock.locked():
            return
        self.focus_incr()
        self.selector = Lasso(event.inaxes, (event.xdata, event.ydata), self.update_line_from_verts)
        self.image_axes.figure.canvas.widgetlock(self.selector)

    def update_line_from_verts(self, verts):
        current_line = self.lines[self.focus_index]
        current_roi = self.rois[self.focus_index]

        for x,y in verts:
            current_roi.x.append(x)
            current_roi.y.append(y)
        self.image_axes.figure.canvas.widgetlock.release(self.selector)
        current_line.set_data(current_roi.x, current_roi.y)
        current_line.figure.canvas.draw()

    def wipe(self, event):
        current_line = self.lines[self.focus_index]
        current_roi = self.rois[self.focus_index]
        current_roi.reset()
        current_line.set_data(current_roi.x, current_roi.y)
        current_line.figure.canvas.draw()