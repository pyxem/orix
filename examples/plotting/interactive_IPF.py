"""
====================================
Interactive IPF map with Euler angle
====================================

This example shows how to use
:doc:`matplotlib event connections <matplotlib:users/explain/event_handling>`
to add an interactive click function to the inverse pole figure (IPF) map to
retrieve the phase name and corresponding Euler angles from the location of
click.

.. note::
    This example shows the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""
import matplotlib.pyplot as plt
import numpy as np

from orix import data, plot

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

# Add an overlay of dot product to the orientation color map to enhance grain boundary contrast=
xmap_overlay = rgb_all.reshape(xmap.shape + (3,))
overlay_1dim = (xmap.prop["dp"]).reshape(xmap.shape)
overlay_min = np.nanmin(overlay_1dim)
rescaled_overlay = (overlay_1dim - overlay_min) / (
    np.nanmax(overlay_1dim) - overlay_min
)
n_channels = 3
for i in range(n_channels):
    xmap_overlay[:, :, i] *= rescaled_overlay
xmap_image = xmap_overlay


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

        xmap_yx = xmap[int(y_pos), int(x_pos)]
        eu = xmap_yx.rotations.to_euler(degrees=True)[0]
        phase_name = xmap_yx.phases_in_data[:].name
        plt.plot(x_pos, y_pos, "+", c="black", markersize=15, markeredgewidth=3)
        plt.title(f"Phase: {phase_name}, Euler angles: {np.array_str(eu, precision=2)[1:-1]}")
        plt.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.axis("off")
    plt.show()
    plt.draw()
    return coords  # click point coordintes in [x, y] format


result = select_point(xmap_gb)
