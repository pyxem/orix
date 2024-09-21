"""
============================
Interactive crystal map plot
============================

This example shows how to use
:doc:`matplotlib event connections <matplotlib:users/explain/figure/event_handling>` to
add an interactive click function to a :class:`~orix.crystal_map.CrystalMap` plot.
Here, we navigate an inverse pole figure (IPF) map and retreive the phase name and
corresponding Euler angles from the location of the click.

.. note::

    This example uses the interactive capabilities of Matplotlib, and this will not
    appear in the static documentation.
    Please run this code on your machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example using the
    link at the bottom of the page.
"""

import matplotlib.pyplot as plt
import numpy as np

from orix import data, plot
from orix.crystal_map import CrystalMap

xmap = data.sdss_ferrite_austenite(allow_download=True)
print(xmap)

pg_laue = xmap.phases[1].point_group.laue
O_au = xmap["austenite"].orientations
O_fe = xmap["ferrite"].orientations

# Get IPF colors
ipf_key = plot.IPFColorKeyTSL(pg_laue)
rgb_au = ipf_key.orientation2color(O_au)
rgb_fe = ipf_key.orientation2color(O_fe)

# Combine IPF color arrays
rgb_all = np.zeros((xmap.size, 3))
phase_id_au = xmap.phases.id_from_name("austenite")
phase_id_fe = xmap.phases.id_from_name("ferrite")
rgb_all[xmap.phase_id == phase_id_au] = rgb_au
rgb_all[xmap.phase_id == phase_id_fe] = rgb_fe


def select_point(xmap: CrystalMap, rgb_all: np.ndarray) -> tuple[int, int]:
    """Return location of interactive user click on image.

    Interactive function for showing the phase name and Euler angles
    from the click-position.
    """
    fig = xmap.plot(
        rgb_all,
        overlay="dp",
        return_figure=True,
        figure_kwargs={"figsize": (12, 8)},
    )
    ax = fig.axes[0]
    ax.set_title("Click position")

    # Extract array in the plot with IPF colors + dot product overlay
    rgb_dp_2d = ax.images[0].get_array()

    x = y = 0

    def on_click(event):
        x, y = (event.xdata, event.ydata)
        if x is None:
            print("Please click inside the IPF map")
            return
        print(x, y)

        # Extract location in crystal map and extract phase name and
        # Euler angles
        xmap_yx = xmap[int(np.round(y)), int(np.round(x))]
        phase_name = xmap_yx.phases_in_data[:].name
        eu = xmap_yx.rotations.to_euler(degrees=True)[0].round(2)

        # Format Euler angles
        eu_str = "(" + ", ".join(np.array_str(eu)[1:-1].split()) + ")"

        plt.clf()
        plt.imshow(rgb_dp_2d)
        plt.plot(x, y, "+", c="k", markersize=15, markeredgewidth=3)
        plt.title(
            rf"Phase: {phase_name}, Euler angles: $(\phi_1, \Phi, \phi_2)$ = {eu_str}"
        )
        plt.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)

    return x, y


x, y = select_point(xmap, rgb_all)
plt.show()
