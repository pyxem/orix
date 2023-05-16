"""
========
Interactive IPF map with Euler angle 
========

This example shows how to use [`matplotlib event connections`](https://matplotlib.org/stable/users/explain/event_handling.html) 
to add an interactive click function to the IPF plot to retrieve the phase 
name and corresponding Euler angles from the location of click.
"""
import matplotlib.pyplot as plt
import numpy as np

from orix import data, plot
from orix.quaternion import Rotation
from orix.vector import Miller

xmap = data.sdss_ferrite_austenite(allow_download=True)
print(xmap)

pg_laue = xmap.phases[1].point_group.laue
ori_au = xmap["austenite"].orientations
ori_fe = xmap["ferrite"].orientations

# Orientation colors
ipf_key = plot.IPFColorKeyTSL(pg_laue)
rgb_au = ipf_key.orientation2color(ori_au)
rgb_fe = ipf_key.orientation2color(ori_fe)

rgb_all = np.zeros((xmap.size, 3))
rgb_all[xmap.phase_id == 1] = rgb_au
rgb_all[xmap.phase_id == 2] = rgb_fe
xmap_gb = rgb_all.reshape(xmap.shape + (3,))


# An interactive function for getting the phase name and euler angles from the clicking position
def select_point(image):
    """Return location of interactive user click on image."""
    fig, ax = plt.subplots(subplot_kw=dict(projection="plot_map"), figsize=(12, 8))
    ax.imshow(image)
    ax.set_title("Click position")
    coords = []

    def on_click(event):
        print(event.xdata, event.ydata)
        coords.append(event.xdata)
        coords.append(event.ydata)
        plt.clf()
        plt.imshow(image)
        try:
            x_pos = coords[-2]
            y_pos = coords[-1]
        except:
            x_pos = 0
            y_pos = 0

        phase = xmap.phases[xmap[int(y_pos), int(x_pos)].phase_id[0]].name
        [Eu1, Eu2, Eu3] = np.rad2deg(
            Rotation.to_euler(xmap[int(y_pos), int(x_pos)].orientations)
        )[0]
        plt.plot(x_pos, y_pos, "+", c="black", markersize=15, markeredgewidth=3)
        plt.title(
            f"Phase: {phase}, Euler angles: {np.round(Eu1, 2)}, {np.round(Eu2, 2)}, {np.round(Eu3, 2)}"
        )
        plt.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    plt.draw()
    return coords  # click point coordintes in [x, y] format


result = select_point(xmap_gb)
