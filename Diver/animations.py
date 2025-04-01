"""
module for animating the DE algorithm
"""

import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import PillowWriter, FuncAnimation


from de import diver, parabolic, two_valleys, four_valleys


"""
global constants
"""
NP = 50
F = 0.7
Cr = 0.7
ranges = [(-50, 50)] * 2

conv_thresh = 1e-5
MAX_ITER = 10000

obj_func = four_valleys

animation_name = "four_valleys"
cont_levels = 10


"""
create data
"""
populations, improvements, update_times = diver(NP, ranges, F, Cr, obj_func, conv_thresh, MAX_ITER)


"""custom colormap for histograms"""


def hist_cmap():
    """
    custom color map
    """
    base = mpl.colormaps.get_cmap("plasma")
    colors = [(1, 1, 1, 1)] + [base(i / (base.N - 1)) for i in range(1, base.N)]
    cmap = mpl.colors.ListedColormap(colors)
    return cmap


"""
start animation
"""


def animate() -> None:
    """
    create a 3d represantation of the objective function and animate the different populations in a contour plot as well as a 2d density histogram
    """
    # define metadata and writer
    metadata = dict(title="diver animation", artist="Valentin Reichenspurner")
    writer = PillowWriter(fps=15, metadata=metadata)

    # create meshgrid
    y = list(np.linspace(ranges[1][0], ranges[1][1], 100))
    x = list(np.linspace(ranges[0][0], ranges[0][1], 100))
    X, Y = np.meshgrid(x, y)

    # vectorize objective function
    obj_func_vectorized = np.vectorize(lambda x, y: obj_func([x, y]))

    # evaluate objective function
    Z = obj_func_vectorized(X, Y)

    # create figure
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221, projection="3d")
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)

    # create 3D surface plot for ax1
    ax1.plot_surface(X, Y, Z, cmap="plasma")
    ax1.set_axis_off()

    # add contours for ax2
    contour = ax2.contour(X, Y, Z, levels=cont_levels)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(ranges[0][0], ranges[0][1])
    ax2.set_ylim(ranges[1][0], ranges[1][1])
    ax2.set_aspect("equal", adjustable="box")
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(contour, label="p(x,y)", ax=ax2, cax=cax2)
    scatter = ax2.scatter([], [], marker=".", color="magenta", alpha=0.8, zorder=100)

    # create 2d histogram for ax3
    point_collection = list(itertools.chain(*populations))
    points_x, points_y = zip(*point_collection)
    step_size = 5
    bins_x = int(abs(ranges[0][1] - ranges[0][0]) / step_size)
    bins_y = int(abs(ranges[1][1] - ranges[1][0]) / step_size)

    hist, xedges, yedges = np.histogram2d(points_x, points_y, bins=[bins_x, bins_y])
    print(np.max(hist))

    cmap = hist_cmap()
    norm = mpl.colors.Normalize(vmin=1, vmax=np.max(hist))
    div3 = make_axes_locatable(ax3)
    cax3 = div3.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), label="count", ax=ax3, cax=cax3)

    im = ax3.imshow(hist, cmap=cmap, norm=norm, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin="lower")
    #   ax3.hist2d(points_x, points_y, bins=[bins_x, bins_y], cmap=cmap)
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_xlim(ranges[0][0], ranges[0][1])
    ax3.set_ylim(ranges[1][0], ranges[1][1])
    ax3.set_aspect("equal", adjustable="box")
    # calculate frames
    hist_frames = []
    for frame in range(len(populations)):
        points_x, points_y = zip(*list(itertools.chain(*populations[: frame + 1])))
        hist, xedges, yedges = np.histogram2d(points_x, points_y, bins=[bins_x, bins_y])
        hist_frames.append(hist.T)

    def update(frame):

        im.set_data(hist_frames[frame])
        scatter.set_offsets(populations[frame])
        return scatter, im

    ani = FuncAnimation(fig, update, frames=len(populations), interval=100, blit=True)
    ani.save("./animations/" + animation_name + ".gif", writer=writer)

    # select animation frames and write gif


#   with writer.saving(fig, './animations/' + animation_name + '.gif', dpi=100):
#       for id, frame in enumerate(populations):
#           points_x, points_y = zip(*list(itertools.chain(*populations[:id+1])))
#           Hist, xedges, yedges = np.histogram2d(points_x, points_y, bins=[bins_x, bins_y])
#           im.set_data(Hist.T)
#           im.set_extent([xedges[0], xedges[-1], yedges[0], yedges[-1]])
#           writer.grab_frame()

#
#           scatter.set_offsets(frame)

#           writer.grab_frame()


if __name__ == "__main__":
    animate()
