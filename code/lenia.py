import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from movie import *

import species

# Utils

path_simul = (
    "/Users/arsnm/Documents/cpge/mp2/tipe-mp2/simul/"  # absolute path ! careful !
)
path_graphs = "/Users/arsnm/Documents/cpge/mp2/tipe-mp2/doc/slideshow/img/"


def gauss(x, mu: float, sigma: float):
    """Return non-normalized gaussian function of expected value mu and
    variance sigma ** 2"""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def polynomial(x, alpha: int):
    return (4 * x * (1 - x)) ** alpha


# Game of life (GoL)


def evolution_gol(grid):
    # count the neighbor considering a periodic grid (wrappred around its border)
    neighbor_count = sum(
        np.roll(np.roll(grid, i, 0), j, 1)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        if (i != 0 or j != 0)
    )
    return (neighbor_count == 3) | (grid & (neighbor_count == 2))


# simulation test
grid = np.random.randint(0, 2, (64, 64))
# create_movie(
#     grid,
#     evolution_gol,
#     path_simul + "gol_simul.mp4",
#     200,
#     cmap="plasma",
#     interpolation="none",
#     interval=200,
# )

# GoL with continuous kernel and growth

kernel_gol = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def growth_gol(neighbor_val):
    cond1 = (neighbor_val >= 1) & (neighbor_val <= 3)
    cond2 = (neighbor_val > 3) & (neighbor_val <= 4)
    return -1 + (neighbor_val - 1) * cond1 + 8 * (1 - neighbor_val / 4) * cond2


def evolution_continuous_gol(grid):
    neighbor_count = sp.signal.convolve2d(
        grid, kernel_gol, mode="same", boundary="wrap"
    )
    grid = grid + growth_gol(neighbor_count)
    grid = np.clip(grid, 0, 1)
    return grid


# simulation test
grid = np.random.randint(0, 2, (64, 64))
# create_movie(
#     grid,
#     evolution_continuous_gol,
#     path_simul + "gol_continuous_simul.mp4",
#     200,
#     cmap="plasma",
#     interpolation="none",
#     interval=200,
# )

# Lenia

scale = 1  # scaling factor to speed up rendering when testing

# Ring filter

R = 13  # radius of kernel
x, y = np.ogrid[-R:R, -R:R]  # grid
dist_norm = (((1 + x) ** 2 + (1 + y) ** 2) ** 0.5) / R  # normalized so that dist(R) = 1

gamma = 0.5
delta = 0.15
kernel_shell = (dist_norm <= 1) * gauss(
    dist_norm, gamma, delta
)  # we don't consider neighbor having dist > 1
kernel_shell = kernel_shell / np.sum(kernel_shell)  # normalizing values

# show ring
plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.imshow(dist_norm, interpolation="none", cmap="plasma")
plt.subplot(122)
plt.imshow(kernel_shell, interpolation="none", cmap="plasma")
plt.savefig(path_simul + "ring_kernel.png")

# Growth function


def growth_lenia(region):
    mu = 0.15
    sigma = 0.015
    return -1 + 2 * gauss(region, mu, sigma)


# Evolve function

dt = 0.1  # set the time step


def evolution_lenia(grid):
    neighbor = sp.signal.convolve2d(grid, kernel_shell, mode="same", boundary="wrap")
    grid = grid + dt * growth_lenia(neighbor)
    grid = np.clip(grid, 0, 1)
    return grid


# simulation test
size = int(256 * scale)
mid = size // 2
grid = np.ones((size, size))

# gaussian spot initialization
radius = int(36 * scale)
y, x = np.ogrid[-mid:mid, -mid:mid]
grid = np.exp(-0.5 * (x**2 + y**2) / radius**2)

# create_movie(grid, evolution_lenia, path_simul + "lenia_spot.mp4", 700, cmap="plasma")


# Graphs

# basic example

# random initialization
grid = np.random.rand(size, size)


def plot_basic_lenia():
    global grid
    fig, ax = plt.subplots(3, 4)
    step = 500
    plotted_steps = [0, 1, 3, 5, 9, 15, 30, 60, 90, 125, 200, 500]
    k = 0
    i, j = 0, 0
    while k <= step:
        if k in plotted_steps:
            ax[i, j].imshow(grid, cmap="plasma")
            ax[i, j].set_title(f"t = {k}", fontweight="bold", fontsize=7)
            ax[i, j].axis("off")
            if j == 3:
                i += 1
                j = 0
            else:
                j += 1
        grid = evolution_lenia(grid)
        k += 1

    fig.tight_layout()
    plt.savefig(path_graphs + "evolution_lenia_random_init.png", transparent=True)


# plot_basic_lenia()


# example with perturbation

# random initialization
grid = np.random.rand(size, size)


def plot_basic_lenia_with_perturbation():
    global grid
    fig, ax = plt.subplots(3, 4)
    step = 500
    plotted_steps = [0, 15, 30, 60, 125, 199, 200, 205, 210, 220, 250, 500]
    step_perturbation = 200
    k = 0
    i, j = 0, 0
    while k <= step:
        if k == step_perturbation:
            for x in range(1 * len(grid) // 3, 2 * len(grid) // 3):
                for y in range(3 * len(grid[i]) // 6, 5 * len(grid[i]) // 6):
                    grid[x, y] = 0
        if k in plotted_steps:
            ax[i, j].imshow(grid, cmap="plasma")
            if k == step_perturbation:
                ax[i, j].set_title(
                    f"t = {k} : Perturbation", color="r", fontweight="bold", fontsize=7
                )
            else:
                ax[i, j].set_title(f"t = {k}", fontweight="bold", fontsize=7)
            ax[i, j].axis("off")
            if j == 3:
                i += 1
                j = 0
            else:
                j += 1

        grid = evolution_lenia(grid)
        k += 1

    fig.tight_layout()
    plt.savefig(
        path_graphs + "evolution_lenia_random_init_perturbation.png", transparent=True
    )


# plot_basic_lenia_with_perturbation()


# random initialization
grid = np.random.rand(size, size)

# create_movie(
#     grid,
#     evolution_lenia,
#     path_simul + "lenia_random.mp4",
#     300,
#     cmap="plasma",
#     interpolation="none",)

# Orbium (gol's glider "equivalent")

orbium = species.orbium

plt.imshow(orbium.T, cmap="plasma")
plt.savefig(path_simul + "orbium.png")
plt.savefig(path_graphs + "orbium.png")

size = 128
grid = np.zeros((size, size))
pos = size // 6
grid[pos : (pos + orbium.shape[1]), pos : (pos + orbium.shape[0])] = orbium.T


def plot_orbium():

    size = 64
    grid_basic = np.zeros((size, size))
    pos = size // 6
    grid_basic[pos : (pos + orbium.shape[1]), pos : (pos + orbium.shape[0])] = orbium.T
    grid_perturbation = grid_basic.copy()

    fig, ax = plt.subplots(2, 6, figsize=(12, 5))
    fontsize = 10

    step = 100
    plotted_steps = [0, 25, 50, 75, 100]
    step_perturbation = 50

    ax[0, 0].imshow(orbium, cmap="plasma", interpolation="bicubic", vmin=0, vmax=1)
    ax[0, 0].set_title("Forme", fontsize="x-large")
    ax[0, 0].axis("off")
    k = 0
    j1, j2 = 1, 0
    while k <= step:
        if k == step_perturbation:
            for x in range(len(grid_perturbation)):
                for y in range(len(grid_perturbation[x])):
                    if grid_perturbation[x, y] > 0:
                        v = np.random.choice(
                            [0, grid_perturbation[x, y]], p=[1 / 3, 1 - 1 / 3]
                        )
                        grid_perturbation[x, y] = v
        if k in plotted_steps:
            ax[0, j1].imshow(grid_basic, cmap="plasma")
            ax[1, j2].imshow(grid_perturbation, cmap="plasma")
            ax[0, j1].axis("off")
            ax[1, j2].axis("off")
            ax[0, j1].set_title(f"t = {k}", fontweight="bold", fontsize=fontsize)
            if k == step_perturbation:
                ax[1, j2].set_title(
                    f"t = {k} : Perturbation",
                    color="r",
                    fontweight="bold",
                    fontsize=fontsize,
                )
            else:
                ax[1, j2].set_title(f"t = {k}", fontweight="bold", fontsize=fontsize)
            j1 += 1
            j2 += 1

        if k == step_perturbation - 1:
            ax[1, j2].imshow(grid_perturbation, cmap="plasma")
            ax[1, j2].axis("off")
            ax[1, j2].set_title(f"t = {k}", fontweight="bold", fontsize=7)
            j2 += 1

        grid_basic = evolution_lenia(grid_basic)
        if k < step_perturbation:
            grid_perturbation = grid_basic.copy()
        else:
            grid_perturbation = evolution_lenia(grid_perturbation)
        k += 1

    fig.tight_layout()
    plt.savefig(path_graphs + "evolution_orbium.png", transparent=True)
    plt.clf()


# plot_orbium()


# create_movie(
#     grid, evolution_lenia, path_simul + "lenia_orbium.mp4", 100, cmap="plasma", interval=50
# )

# Lenia optimization with fft

size = 128
mid = size // 2
grid = np.zeros((size, size))
pos = size // 6
grid[pos : (pos + orbium.shape[1]), pos : (pos + orbium.shape[0])] = orbium.T

# redefine kernel to meet fft's requirements

R = 13
x, y = np.ogrid[-mid:mid, -mid:mid]  # grid
dist_norm = (((x**2 + y**2) ** 0.5)) / R  # normalized so that dist(R) = 1
kernel_shell = (dist_norm <= 1) * gauss(dist_norm, 0.5, 0.15)
kernel_shell = kernel_shell / np.sum(kernel_shell)
f_kernel = sp.fft.fft2(sp.fft.fftshift(kernel_shell))  # fft of kernel
# show ring fft
plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.imshow(dist_norm, interpolation="none", cmap="plasma")
plt.subplot(122)
plt.imshow(kernel_shell, interpolation="none", cmap="plasma")
plt.savefig(path_simul + "ring_kernel_fft.png")
plt.clf()


def evolution_lenia_fft(grid):
    neighbor = np.real(sp.fft.ifft2(f_kernel * sp.fft.fft2(grid)))
    grid = np.clip(grid + dt * growth_lenia(neighbor), 0, 1)
    return grid


# create_movie(
#     grid,
#     evolution_lenia_fft,
#     path_simul + "lenia_orbium_fft.mp4",
#     500,
#     cmap="plasma",
#     interval=50,
# )


# Multi Kernel

size = 128
mid = size // 2
x, y = np.ogrid[-mid:mid, -mid:mid]
R = 36
amplitude = [1, 0.667, 0.333, 0.667]
dist_norm = (x**2 + y**2) ** 0.5 / R * len(amplitude)

kernel_multi_quadrium = np.zeros_like(dist_norm)
alpha = 4
for i in range(len(amplitude)):
    kernel_multi_quadrium += (
        (dist_norm.astype(int) == i) * amplitude[i] * polynomial(dist_norm % 1, alpha)
    )
kernel_multi_quadrium /= np.sum(kernel_multi_quadrium)
f_kernel = sp.fft.fft2(sp.fft.fftshift(kernel_multi_quadrium))  # fft of kernel


def growth_quadrium_lenia(region):
    mu = 0.16
    sigma = 0.01
    return 2 * gauss(region, mu, sigma) - 1


def evolution_quadrium_fft(grid):
    neighbor = np.real(sp.fft.ifft2(f_kernel * sp.fft.fft2(grid)))
    grid = np.clip(grid + dt * growth_quadrium_lenia(neighbor), 0, 1)
    return grid


grid = np.zeros((size, size))
quadrium = species.quadrium
pos = size // 10
grid[size - quadrium.shape[0] - pos : size - pos, 0 : quadrium.shape[1]] = quadrium

# create_movie(
#     grid,
#     evolution_quadrium_fft,
#     path_simul + "quadrium_simul.mp4",
#     200,
#     cmap="plasma",
#     interpolation="none",
#     interval=200,
# )


def plot_quadrium():
    global grid
    fig, ax = plt.subplots(2, 4)
    step = 500
    plotted_steps = [0, 50, 100, 200, 300, 400, 500]
    k = 0
    i, j = 0, 1
    ax[0, 0].imshow(quadrium, cmap="plasma")
    ax[0, 0].axis("off")
    ax[0, 0].set_title("Forme", fontsize="x-large")
    while k <= step:
        if k in plotted_steps:
            ax[i, j].imshow(grid, cmap="plasma")
            ax[i, j].set_title(f"t = {k}", fontweight="bold", fontsize=7)
            ax[i, j].axis("off")
            if j == 3:
                i += 1
                j = 0
            else:
                j += 1
        grid = evolution_quadrium_fft(grid)
        k += 1

    fig.tight_layout()
    plt.savefig(path_graphs + "evolution_quadrium.png", transparent=True)


# plot_quadrium()

# Multi-channel Lenia

kernels_table = species.aquarium["kernels"]

betas = [k["b"] for k in kernels_table]
mus = [k["m"] for k in kernels_table]
sigmas = [k["s"] for k in kernels_table]
heights = [k["h"] for k in kernels_table]
radii = [k["r"] for k in kernels_table]
sources = [k["c0"] for k in kernels_table]
destinations = [k["c1"] for k in kernels_table]

gamma = 0.5
delta = 0.15

dt = 0.5
R = 12
size = 128
mid = size // 2
x, y = np.ogrid[-mid:mid, -mid:mid]


kernel_shells = []
kernels_fft = []

fig, ax = plt.subplots(1)

for b, r in zip(betas, radii):
    r *= R
    dist_norm = (x**2 + y**2) ** 0.5 / r * len(b)
    kernel_shell = np.zeros_like(dist_norm)
    for i in range(len(b)):
        mask_norm = dist_norm.astype(int) == i
        kernel_shell += mask_norm * b[i] * gauss(dist_norm % 1, gamma, delta)
    kernel_shell /= kernel_shell.sum()
    kernel_shells.append(kernel_shell)
    kernels_fft.append(sp.fft.fft2(sp.fft.fftshift(kernel_shell)))

colors = {0: "r", 1: "g", 2: "b"}
for i, k in enumerate(kernel_shells):
    ax.plot(k[size // 2, :], color=colors[sources[i]], label=f"canal {sources[i]}")
    ax.xaxis.set_visible(False)

fig.tight_layout()
# plt.savefig(path_graphs + "plot_kernel_multi_channel.png", transparent=True)

grids = [np.zeros((size, size)) for _ in range(3)]


def evolution_multi_channel(grids):
    grids_fft = [sp.fft.fft2(grid) for grid in grids]
    potentials = [
        np.real(np.fft.ifft2(kernel_fft * grids_fft[source]))
        for kernel_fft, source in zip(kernels_fft, sources)
    ]
    growths_potential = [
        2 * gauss(potential, mus[i], sigmas[i]) - 1
        for i, potential in enumerate(potentials)
    ]
    growths = np.zeros_like(grids)
    for destination, height, growth in zip(destinations, heights, growths_potential):
        growths[destination] += height * growth
    grids = [np.clip(grid + dt * growth, 0, 1) for grid, growth in zip(grids, growths)]
    return grids


aquarium = [np.array(species.aquarium["cells"][c]) for c in range(3)]

for c in range(3):
    grids[c][mid : mid + aquarium[c].shape[0], mid : mid + aquarium[c].shape[1]] = (
        aquarium[c]
    )


def plot_aquarium():
    global grids
    fig, ax = plt.subplots(3, 4)
    step = 600
    plotted_steps = [0, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600]
    k = 0
    i, j = 0, 1
    ax[0, 0].imshow(np.dstack(aquarium))
    ax[0, 0].axis("off")
    ax[0, 0].set_title("Forme", fontsize="x-large")
    while k <= step:
        if k in plotted_steps:
            ax[i, j].imshow(np.dstack(grids))
            ax[i, j].set_title(f"t = {k}", fontweight="bold", fontsize=7)
            ax[i, j].axis("off")
            if j == 3:
                i += 1
                j = 0
            else:
                j += 1
        grids = evolution_multi_channel(grids)
        k += 1

    fig.tight_layout()
    plt.savefig(path_graphs + "evolution_aquarium.png", transparent=True)


# plot_aquarium()
