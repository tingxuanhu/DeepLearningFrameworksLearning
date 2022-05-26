import sys
from IPython import display

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

config = sys.modules[__name__]


def use_svg_display():
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for the matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

    if legend is not None:
        axes.legend(legend)
    axes.grid()


class Animator:
    """For plotting data in animation"""

    def __init__(self,
                 xlabel=None,
                 ylabel=None,
                 legend=None,
                 xlim=None,
                 ylim=None,
                 xscale='linear',
                 yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'),  # format setting
                 nrows=1,
                 ncols=1,
                 figsize=(3.5, 2.5)
                 ):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []

        config.use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)

        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        # use a lambda function to capture arguments
        self.config_axes = lambda: config.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)

        if not hasattr(x, "__len__"):
            x = [x] * n

        if not self.X:
            self.X = [[] for _ in range(n)]

        if not self.Y:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()  # clear the axes

        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

        display.display(self.fig)
        display.clear_output(wait=True)


def train_visualization(num_epochs):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[.3, .9],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        pass


































