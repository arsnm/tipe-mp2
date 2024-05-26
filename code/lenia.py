import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import species
from movie import *

# Utils

path = "/Users/arsnm/Documents/cpge/mp2/tipe-mp2/simul/"  # absolute path ! careful !


def gauss(x, mu: float, sigma: float):
    """Return non-normalized gaussian function of expected value mu and
    variance sigma ** 2"""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


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
#     path + "gol_simul.mp4",
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
#     path + "gol_continuous_simul.mp4",
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

mu = 0.5
sigma = 0.15
kernel = (dist_norm <= 1) * gauss(
    dist_norm, mu, sigma
)  # we don't consider neighbor having dist > 1
kernel = kernel / np.sum(kernel)  # normalizing values

# show ring
plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.imshow(dist_norm, interpolation="none", cmap="plasma")
plt.subplot(122)
plt.imshow(kernel, interpolation="none", cmap="plasma")
plt.savefig(path + "ring_kernel.png")

# Growth function


def growth_lenia(region):
    mu = 0.15
    sigma = 0.015
    return -1 + 2 * gauss(region, mu, sigma)


# Evolve function

dt = 0.1  # set the time step


def evolution_lenia(grid):
    neighbor = sp.signal.convolve2d(grid, kernel, mode="same", boundary="wrap")
    grid = grid + dt * growth_lenia(neighbor)
    grid = np.clip(grid, 0, 1)
    return grid


# simulation test
size = int(512 * scale)
mid = size // 2
grid = np.ones((size, size))

# gaussian spot initialization
radius = int(36 * scale)
y, x = np.ogrid[-mid:mid, -mid:mid]
grid = np.exp(-0.5 * (x**2 + y**2) / radius**2)

# create_movie(grid, evolution_lenia, path + "lenia_spot.mp4", 700, cmap="plasma")

# random initialization
grid = np.random.rand(size, size)

# create_movie(
#     grid,
#     evolution_lenia,
#     path + "lenia_random.mp4",
#     300,
#     cmap="plasma",
#     interpolation="none",
# )

# Orbium (gol's glider "equivalent")

orbium = species.orbium

plt.imshow(orbium, cmap="plasma", interpolation="bicubic", vmin=0, vmax=1)
plt.savefig(path + "orbium.png")

size = 128
grid = np.zeros((size, size))
pos = size // 6
grid[pos : (pos + orbium.shape[1]), pos : (pos + orbium.shape[0])] = orbium.T

create_movie(
    grid, evolution_lenia, path + "lenia_orbium.mp4", 500, cmap="plasma", interval=50
)

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
kernel = (dist_norm <= 1) * gauss(dist_norm, 0.5, 0.15)
kernel = kernel / np.sum(kernel)
f_kernel = sp.fft.fft2(sp.fft.fftshift(kernel))  # fft of kernel
# show ring fft
plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.imshow(dist_norm, interpolation="none", cmap="plasma")
plt.subplot(122)
plt.imshow(kernel, interpolation="none", cmap="plasma")
plt.savefig(path + "ring_kernel_fft.png")


def evolution_lenia_fft(grid):
    neighbor = np.real(sp.fft.ifft2(f_kernel * sp.fft.fft2(grid)))
    grid = np.clip(grid + dt * growth_lenia(neighbor), 0, 1)
    return grid


create_movie(
    grid,
    evolution_lenia_fft,
    path + "lenia_orbium_fft.mp4",
    500,
    cmap="plasma",
    interval=50,
)
