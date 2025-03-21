import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from collections.abc import Callable

from de import diver, parabolic


"""
global constants
"""
NP = 10
F = 0.5
Cr = 0.5
ranges = [(-50,50)]*2

conv_thresh = 1e-5
MAX_ITER = 10000

obj_func = parabolic

animation_name = "parabolic"


"""
create data
"""
populations, improvements = diver(NP, ranges, F, Cr, obj_func)


"""
start animation
"""
def animate() -> None:
    # define metadata and writer
    metadata = dict(title='diver animation', artist='Valentin Reichenspurner')
    writer = PillowWriter(fps=30, metadata=metadata)

    # create meshgrid
    y = list(np.linspace(ranges[1][0], ranges[1][1], 100))
    x = list(np.linspace(ranges[0][0], ranges[0][1], 100))
    X,Y = np.meshgrid(x,y)

    # vectorize objective function
    obj_func_vectorized = np.vectorize(lambda x,y: obj_func([x,y]))

    # evaluate objective function
    Z = obj_func_vectorized(X,Y)

    # create figure
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # create 3D surface plot for ax1
    ax1.plot_surface(X,Y,Z, cmap='plasma')
    ax1.set_axis_off()

    # add contours for ax2
    contour = ax2.contour(X,Y,Z, levels=50)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_xlim(ranges[0][0], ranges[0][1])
    ax2.set_ylim(ranges[1][0], ranges[1][1])
    ax2.set_aspect('equal', adjustable='box')
    fig.colorbar(contour, label="p(x,y)")

    # create empty scatter plot
    scatter = ax2.scatter([], [], marker='.', color='magenta')

    # select animation frames and write gif
    with writer.saving(fig, './animations/' + animation_name + '.gif', dpi=100):
        for frame in populations:
            scatter.set_offsets(frame)
            writer.grab_frame()




if __name__ == "__main__":
    animate()