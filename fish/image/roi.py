class ROI(object):
    """class for representing a single polygonal ROI"""
    def __init__(self, image=[], x=[], y=[]):
        if image is not None:
            self.image = image
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y

    def __repr__(self):
        return 'An ROI containing {0} points'.format(len(self.x))

    def reset(self):
        self.x = []
        self.y = []

    def get_mask(self):
        from matplotlib.path import Path
        from numpy import meshgrid, zeros, array, where
        data = self.image
        mask = zeros(data.shape, dtype='uint8')

        if (len(self.y) > 2) & (len(self.x) > 2):

            points = list(zip(self.y, self.x))

            grid = meshgrid(range(data.shape[0]), range(data.shape[1]))
            coords = list(zip(grid[0].ravel(), grid[1].ravel()))

            path = Path(points)
            in_points = array(coords)[where(path.contains_points(coords))[0]]
            in_points = [tuple(x) for x in in_points]
            for t in in_points:
                mask[t] = 255
        else:
            print('Mask requires 3 or more points')

        return mask


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
        self.rois.append(ROI(image = self.image_data, x=[], y=[]))
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
        current_mask = self.masks[self.focus_index]
        current_roi.reset()
        current_mask = None;
        current_line.set_data(current_roi.x, current_roi.y)
        current_line.figure.canvas.draw()