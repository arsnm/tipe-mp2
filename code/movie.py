import time as t

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

DPI = 100


def create_movie(
    X, evolve, path, steps=100, cmap=None, interpolation="bicubic", interval=50
):

    print(f"Rendering {path}")
    time = t.time()
    if len(X.shape) == 2 and cmap is None:
        cmap = "gray_r"

    fig = plt.figure(figsize=(16, 12))
    im = plt.imshow(X, cmap=cmap, interpolation=interpolation, vmin=0, vmax=1)
    plt.axis("off")

    def update(i):

        if i % (steps // 10) == 0:
            print(f"Step {i}/{steps}")

        if i == 0:
            return (im,)
        nonlocal X
        X = evolve(X)
        im.set_array(X)
        return (im,)

    ani = anim.FuncAnimation(fig, update, steps, interval=interval, blit=True)
    ani.save(path, fps=25, dpi=DPI)
    time = t.time() - time
    print(f"Done in {time//60}min{time%60}s")


def create_movie_multi(Xs, evolve, path, steps=100, interpolation="bicubic"):

    fig = plt.figure(figsize=(16, 9))
    im = plt.imshow(np.dstack(Xs), interpolation=interpolation)
    plt.axis("off")

    def update(i):

        if i % (steps // 10) == 0:
            print(f"Step {i}/{steps}")

        if i == 0:
            return (im,)
        nonlocal Xs
        Xs = evolve(Xs)
        im.set_array(np.dstack(Xs))
        return (im,)

    ani = anim.FuncAnimation(fig, update, steps, interval=50, blit=True)
    ani.save(path, fps=25, dpi=DPI)
