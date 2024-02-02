import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.animation import FuncAnimation


# utils
def figure_world(A, cmap="viridis"):
    global img
    fig = plt.figure()
    img = plt.imshow(A, cmap=cmap, interpolation="nearest", vmin=0)
    plt.title("world A")
    plt.close()
    return fig


size = 64  # define the size of the grid

cells = np.random.randint(2, size=(size, size))  # initiate cells
convolution_kernel = np.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
kernel_sum = np.sum(convolution_kernel)


def update(frame):
    global cells
    neighbour = sp.signal.convolve2d(
        cells, convolution_kernel, mode="same", boundary="wrap"
    )
    cells = (cells & neighbour == 2) | neighbour == 3
    img.set_array(cells)
    return (img,)


fig = figure_world(cells, cmap="binary")

anim = FuncAnimation(fig, func=update, frames=50, interval=50)
plt.show()
