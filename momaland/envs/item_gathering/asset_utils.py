"""disgracefully stolen.

https://stackoverflow.com/questions/73956440/python-to-change-image-into-a-color-that-is-not-one-of-the-images-dominant-colo
"""

import os
import shutil

import numpy as np
from PIL import Image


def get_colored(asset: str, count: int):
    """Generating random colored visual assets for dynamic environment variables.

    Args:
        asset: `str` value. Either "item" or "agent".
        count: `int` value. The amount of total agents/items in the environment.

    Returns:
        numpy.ndarray with RGB values for the new colored assets.
    """
    assert asset == "item" or asset == "agent", "You must specify an asset to get colored versions of."
    path = os.path.dirname(os.path.realpath(__file__))
    seed = 1 if asset == "item" else 2
    rng = np.random.default_rng(seed)  # local seed
    im = Image.open(f"{path}/assets/{asset}.png").convert("RGBA")
    alpha = im.getchannel("A")
    im = im.convert("RGB")

    for i in range(count - 1):  # subtracting the default asset
        rgb = rng.random((3, 3))
        matrix = (rgb[0, 0], rgb[1, 0], rgb[2, 0], 0, rgb[0, 1], rgb[1, 1], rgb[2, 1], 0, rgb[0, 2], rgb[1, 2], rgb[2, 2], 0)

        # Apply above transform, reinsert alpha and save
        colored_im = im.convert("RGB", matrix)
        colored_im.putalpha(alpha)
        if not os.path.isdir(f"{path}/assets/colored/"):
            os.mkdir(f"{path}/assets/colored/")
        colored_im.save(f"{path}/assets/colored/{asset}{i}.png")


def del_colored():
    """Deleting the previously random colored visual assets."""
    path = os.path.dirname(os.path.realpath(__file__))
    shutil.rmtree(f"{path}/assets/colored/")  # deletes the leaf directory and its content
