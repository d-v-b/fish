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


def proj_fuse(data, fun, aspect=(1, 1, 1), fill_value=0):
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


    stretched[:new_dims[1], new_dims[2]:] = projs[2].T
    stretched[new_dims[1]:, :new_dims[2]] = projs[1]
    stretched[:new_dims[1], :new_dims[2]] = projs[0]

    return stretched


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