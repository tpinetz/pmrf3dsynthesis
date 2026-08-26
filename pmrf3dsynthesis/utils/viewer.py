from __future__ import annotations

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from typing import List, Union

import ipywidgets as widgets
from IPython.display import display


from pmrf3dsynthesis.visualize_eval import EvaluationViewer

__all__ = ['SliceViewer', 'EvaluationViewer']


def xaligned_axes(ax, y_distance, width, **kwargs):
    return plt.axes([
        ax.get_position().x0,
        ax.get_position().y0-y_distance,
        ax.get_position().width, width
    ], **kwargs)


def show_list_of_scans_in_grid(
    scans: list,
    h_w: tuple = None,
    v_int: list = None
):
    if h_w is None:
        h_w = (int(np.ceil(len(scans) / 4)), min(4, len(scans)))

    if v_int is None:
        v_int = [(scan.min(), scan.max()) for scan in scans]

    fig, ax = plt.subplots(h_w[0], h_w[1], sharex=True, sharey=True)
    if len(scans) > 1:
        ax = ax.ravel()
    else:
        ax = [ax]
    viewers = []
    for i, scan in enumerate(scans):
        viewers.append(SliceViewer(ax[i], scan, cmap='gray',
                                   vmin=v_int[i][0],
                                   vmax=v_int[i][1]))

    viewers[0].connect(viewers)
    return viewers


class SliceViewer:
    def __init__(
        self,
        ax: np.ndarray,
        volume: np.ndarray,
        slice: int = None,
        label: np.array = None,
        overlay: np.array = None,
        alpha: float = 0.5,
        **kwargs
    ):
        self._ax = ax

        self._v = volume
        self._l = label if label else overlay
        self._ind = self._v.shape[0] // 2 if slice is None else slice

        self._im = ax.imshow(self._v[self._ind], **kwargs)
        if label is not None:
            cmap = cm.get_cmap("tab10", 10)
            cmap.colors[:, -1] = alpha
            cmap.colors[0, -1] = 0
            self._ov = ax.imshow(self._l[self._ind], cmap=cmap, vmin=0, vmax=10)
        elif overlay is not None:
            self._ov = ax.imshow(self._l[self._ind], cmap="magma", alpha=alpha, vmin=overlay.min(), vmax=overlay.max())
        else:
            self._ov = None
        ax.axis('off')

        self._slider = None
        # self.update({"new": self._ind})

    def slider(self):
        self._slider = widgets.IntSlider(self._ind, min=0, max=self._v.shape[0], description="Slice:")
        display(self._slider)

        self.speed = 1

        self._slider.observe(self.update, names="value")
        return self

    def connect(self, other: Union[SliceViewer, List[SliceViewer]], speed: float = 1.):
        if self._slider is None:
            self._slider = widgets.IntSlider(self._ind, min=0, max=self._v.shape[0], description="Slice:")
            display(self._slider)

            self._slider.observe(self.update, names="value")
        if isinstance(other, SliceViewer):
            other = [other]
        elif isinstance(other, list):
            pass
        else:
            raise RuntimeError("Unsupport type for other!")

        self.speed = 1.

        for o in other:
            if o != self:
                o._slider = self._slider
                o.speed = o._v.shape[0]/self._v.shape[0]
                self._slider.observe(o.update, names="value")
        return self

    def update(self, change):
        val = int(change["new"] * self.speed)
        self._ind = val
        self._im.set_data(self._v[min(len(self._v) - 1, self._ind)])
        if self._ov is not None:
            self._ov.set_data(self._l[min(len(self._l) - 1, self._ind)])


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 3)

    X = np.random.rand(20, 20, 40)
    X1 = np.random.rand(20, 20, 40)

    tracker = SliceViewer(ax[0], X, slice=2, cmap='gray')
    tracker1 = SliceViewer(ax[1], X1)
    tracker2 = SliceViewer(ax[2], X1)

    tracker.connect([tracker1, tracker2], 5)
    # tracker.connect(tracker2)

    plt.show()
