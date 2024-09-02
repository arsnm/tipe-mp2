# Rendering the different graphs used for the project
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from PIL import Image

path = "/Users/arsnm/Documents/cpge/mp2/tipe-mp2/doc/slideshow/img/"


## Game of Life

# evolution step

gol_cmap = colors.ListedColormap(["#960c6b", "#000066", "#cdd300"])
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = colors.BoundaryNorm(bounds, gol_cmap.N)


def evolution_gol(grid, step=1):
    assert step >= 0, "step argument must be >= 0"
    evolved = grid.copy()
    evolved = evolved.astype(int)
    for _ in range(step):
        # count the neighbor considering a periodic grid (wrappred around its border)
        neighbor_count = sum(
            np.roll(np.roll(evolved, i, 0), j, 1)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            if (i != 0 or j != 0)
        )
        evolved = (neighbor_count == 3) | (evolved & (neighbor_count == 2))
    return evolved


# simulation test
np.random.seed(69)
grid = np.random.choice([0, 1], (64, 64), True, p=[0.8, 0.2])
evolved = evolution_gol(grid)
grid[42, 5] = -1
evolved[42, 5] = -1

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(grid, interpolation="none", cmap=gol_cmap, norm=norm)
axs[0].tick_params(which="minor", bottom=False, left=False)
axs[0].invert_yaxis()
axs[0].set_title("Situation Initiale", fontsize=20)


axs[1].imshow(evolved, interpolation="none", cmap=gol_cmap, norm=norm)
axs[1].tick_params(which="minor", bottom=False, left=False)
axs[1].invert_yaxis()
axs[1].set_title("Après une évolution", fontsize=20)

ax_zoom = zoomed_inset_axes(axs[0], zoom=8, loc="upper right")
ax_zoom.imshow(grid, cmap=gol_cmap, norm=norm)

# subregion of the original image
x1, x2, y1, y2 = 3.5, 6.5, 40.5, 43.5
ax_zoom.set_xlim(x1, x2)
ax_zoom.set_ylim(y1, y2)  # fix the number of ticks on the inset Axes
ax_zoom.yaxis.get_major_locator().set_params(nbins=4)
ax_zoom.xaxis.get_major_locator().set_params(nbins=4)
ax_zoom.set_xticks(np.arange(3.5, 7, 1), minor=True)
ax_zoom.set_yticks(np.arange(40.5, 44, 1), minor=True)

ax_zoom.tick_params(labelleft=False, labelbottom=False)
ax_zoom.grid(which="minor", color="black", linewidth=2)

# draw a bbox of the region of the inset Axes in the parent Axes and
# connecting lines between the bbox and the inset Axes area
mark_inset(
    axs[0], ax_zoom, loc1=2, loc2=4, fc="none", ec="0.5", color="red", linewidth=3
)

plt.savefig(path + "plot_evolution_gol.png", transparent=True)
plt.clf()


# species

grid_block = np.zeros((6, 6))
coord_block = [(2, 2), (2, 3), (3, 2), (3, 3)]
for coord in coord_block:
    grid_block[coord] = 1

grid_blinker = np.zeros((5, 5))
coord_blinker = [(2, 1), (2, 2), (2, 3)]
for coord in coord_blinker:
    grid_blinker[coord] = 1

grid_glider = np.zeros((16, 16))
coord_glider = [(1, 1), (2, 2), (2, 3), (3, 1), (3, 2)]
for coord in coord_glider:
    grid_glider[coord] = 1


fig = plt.figure(figsize=(10, 6))
outer_gs = GridSpec(
    2,
    2,
    figure=fig,
    height_ratios=[1, 1],
    width_ratios=[1, 1.5],
    hspace=0.1,
    wspace=0.2,
)


def add_centered_pcolor(sub_gs, data_list, plot_titles, line_title):
    num_plots = len(data_list)
    ax_line = fig.add_subplot(sub_gs)
    ax_line.text(
        0.5,
        1,
        line_title,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        transform=ax_line.transAxes,
    )
    ax_line.axis("off")

    inner_gs = GridSpecFromSubplotSpec(1, num_plots, subplot_spec=sub_gs, wspace=0.1)
    for i, (data, plot_title) in enumerate(zip(data_list, plot_titles)):
        ax = fig.add_subplot(inner_gs[i])
        ax.pcolor(data, cmap="plasma", edgecolor="grey", linewidth=0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(plot_title, fontsize=12)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


data_block = [grid_block, evolution_gol(grid_block)]
data_blinker = [evolution_gol(grid_blinker, i) for i in range(3)]
data_glider = [evolution_gol(grid_glider, 11 * i) for i in range(5)]

titles_block = ["t = 0", "t = 1"]
titles_blinker = ["t = 0", "t = 1", "t = 2"]
titles_glider = [f"t = {11 * i}" for i in range(5)]

plot_block = outer_gs[0, 0]
add_centered_pcolor(plot_block, data_block, titles_block, "Block")

plot_blinker = outer_gs[0, 1]
add_centered_pcolor(plot_blinker, data_blinker, titles_blinker, "Blinker")

plot_glider = outer_gs[1, :]
add_centered_pcolor(plot_glider, data_glider, titles_glider, "Glider")

plt.savefig(path + "plot_species_gol.png", transparent=True)
plt.clf()


# Kernels
def indicator(arr, lower_bound: float = 0, upper_bound: float = 1):
    if type(arr) in ["float", "int"]:
        return int(lower_bound <= arr <= upper_bound)
    else:
        arr = np.copy(arr)
        mask = (arr >= lower_bound) & (arr <= upper_bound)
        arr[mask] = 1
        arr[~mask] = 0
        return arr


def gauss(x, gamma: float = 0.5, delta: float = 0.15):
    return np.exp(-0.5 * ((x - gamma) / delta) ** 2)


def polynomial(x, alpha: int = 4):
    return (4 * x * (1 - x)) ** alpha


fig = plt.figure(figsize=(10, 13))

subfigs = fig.subfigures(1, 2)

dist_1d = np.arange(0, 1, 0.001)
step = 100j
x, y = np.ogrid[-1 : 1 : 2 * step, -1 : 1 : 2 * step]  # grid
dist_norm = ((x) ** 2 + (y) ** 2) ** 0.5

kernel_gauss = (dist_norm <= 1) * (gauss(dist_norm))
kernel_polynomial = (dist_norm <= 1) * (polynomial(dist_norm))
kernel_rectangle = (dist_norm <= 1) * (indicator(dist_norm, 1 / 3, 2 / 3))

ax1 = subfigs[0].subplots(2, 1)
ax1[0].plot(dist_1d, gauss(dist_1d))
ax1[0].text(
    0.5,
    0,
    r"$\gamma = 0.5, \delta = 0.15$",
    fontsize=20,
    horizontalalignment="center",
)
ax1[0].set_xlabel("Distance", fontsize="x-large")
im = ax1[1].imshow(kernel_gauss, interpolation="none", cmap="plasma")
ax1[1].axis("off")
subfigs[0].suptitle("Exponentiel ", fontsize=30, fontweight="bold")

ax2 = subfigs[1].subplots(2, 1)
ax2[0].plot(
    dist_1d,
    indicator(dist_1d, 2 / 4, 3 / 4),
    label=r"$[a, b] = [\frac{2}{4}, \frac{3}{4}]$",
)
ax2[0].text(
    0,
    0.5,
    r"$[a, b] = [\frac{2}{4}, \frac{3}{4}]$",
    fontsize=20,
)
ax2[0].set_xlabel("Distance", fontsize="x-large")
ax2[1].imshow(kernel_rectangle, interpolation="none", cmap="plasma")
ax2[1].axis("off")
subfigs[1].suptitle("Rectangle", fontsize=30, fontweight="bold")

fig.tight_layout(pad=4, h_pad=1, w_pad=4)
fig.subplots_adjust(top=0.92)
cbar_ax = fig.add_axes((0.25, 0.05, 0.5, 0.025))
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
cbar.set_ticks([0, 1])
cbar.ax.tick_params(labelsize=20)


plt.savefig(path + "plot_convolution_kernels.png", transparent=True)
plt.clf()

# Growth mapping

mu1, mu2 = 0.3, 0.7
sigma1, sigma2 = 0.05, 0.2
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
interval = np.arange(0, 1, 0.0001)

axs[0].plot(interval, 2 * gauss(interval, mu1, sigma1) - 1)
axs[0].set_title("Exponentielle", fontsize=20, fontweight="bold")
axs[0].text(
    1,
    0.75,
    f"$\\mu = {mu1}, \\sigma = {sigma1}$",
    fontsize=13,
    horizontalalignment="right",
)

axs[1].plot(interval, 2 * indicator(interval, mu2 - sigma2, mu2 + sigma2) - 1)
axs[1].set_title("Rectangulaire", fontsize=20, fontweight="bold")
axs[1].text(
    0,
    0.75,
    f"$\\mu = {mu2}, \\sigma = {sigma2}$",
    fontsize=13,
    horizontalalignment="left",
)

for i in [0, 1]:
    axs[i].axhline(y=0, color="r", linestyle="--", linewidth=2, alpha=0.5)

plt.savefig(path + "plot_growth_mapping.png", transparent=True)
plt.clf()


# Multi Kernels

A = [0.3, 1, 0.7, 0.2]
gamma = np.random.uniform(0, 1, (len(A),))
delta = np.random.uniform(0, 0.3, (len(A),))
gamma = [0.2, 0.4, 0.6, 0.8]
delta = [0.015, 0.05, 0.01, 0.1]

dist_1d = np.arange(0, 1, 0.001)

step = 1000j
x, y = np.ogrid[-1 : 1 : 2 * step, -1 : 1 : 2 * step]  # grid
dist_norm = ((x) ** 2 + (y) ** 2) ** 0.5

multi_kernel_core = np.zeros_like(dist_1d)
multi_kernel_shell = np.zeros_like(dist_norm)
for i in range(len(A)):
    multi_kernel_core += A[i] * gauss(dist_1d, gamma[i], delta[i])
    multi_kernel_shell += (dist_norm <= 1) * A[i] * gauss(dist_norm, gamma[i], delta[i])


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(dist_1d, multi_kernel_core)
ax[0].set_xlabel("Distance", fontsize="x-large")
im = ax[1].imshow(multi_kernel_shell, cmap="plasma")
ax[1].axis("off")
plt.colorbar(im, ax=ax[1], cmap="plasma", location="bottom", shrink=0.7)

fig.tight_layout()
plt.savefig(path + "plot_multi_ring_kernel.png", transparent=True)


folder_path = (
    "/Users/arsnm/Documents/cpge/mp2/tipe-mp2/simul/resultLeniaMachineLearning/"
)


def name_image(i: int):
    s = str(i + 1)
    while len(s) != 5:
        s = "0" + s
    return s + ".png"


steps = [0, 3, 5, 7, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]
image_names = [name_image(i) for i in steps]

num_images = len(image_names)
cols = 4
rows = (num_images + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.5 * rows))
axes = axes.flatten()

i = 0
for i, image_name in enumerate(image_names):
    img_path = os.path.join(folder_path, image_name)
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(f"t = {steps[i]}", fontweight="bold", fontsize="x-large")
    axes[i].axis("off")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(path + "evolution_machine_learning", transparent=True)
