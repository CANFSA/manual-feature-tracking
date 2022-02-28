# Standard library imports
from pathlib import Path
# Third-party imports
import imageio as iio
import napari
import numpy as np
import pandas as pd
import skimage


def load_images(
    img_dir_path, 
    start=None, 
    stop=None, 
    step=None, 
    manual_img_nums=None, 
    file_suffix='.tif', 
    convert_to_float=True
):
    """Load images from a directory.

    Parameters
    ----------
    img_dir_path : str or Path
        Path to directory containing images to be loaded.
    start : int, optional
        Index of image within directory which will be the first image loaded, by default None
    stop : int, optional
        Index of image within directory which will be the last image loaded, by default None
    step : int, optional
        Step size for images that are loaded between start and stop (e.g. 2 would be every other image), by default None
    manual_img_nums : array-like, optional
        List or array of integers that will be returned as the actual image numbers (only used if image directory is already a subset of the images), by default None
    file_suffix : str, optional
        File suffix for images located in img_dir_path, by default '.tif'
    convert_to_float : bool, optional
        If True, convert images to floating point images, by default True

    Returns
    -------
    tuple
        2-tuple containing np.ndarray of image numbers and list of np.ndarray objects representing images

    Raises
    ------
    ValueError
        If len(manual_img_nums) doesn't match number of loaded images
    """
    img_dir = Path(img_dir_path)
    img_fns = [fn for fn in img_dir.glob(f'*{file_suffix}')]
    if start is None:
        start = 0
    if stop is None:
        # Subtract 1 to account for 1 that is added in img_nums definition
        stop = len(img_fns) - 1
    if step is None:
        step = 1
    # Image range is partially inclusive, so starting point has to be + 1 to include last image
    img_nums = np.arange(start, stop + 1, step)
    imgs = [iio.imread(img_fns[n]) for n in img_nums]
    if convert_to_float:
        imgs = [skimage.util.img_as_float(img) for img in imgs]
    if manual_img_nums is not None:
        img_nums = np.array(manual_img_nums)
    if len(imgs) != len(img_nums):
        raise ValueError(f'Length of manual_img_nums ({len(manual_img_nums)}) must match number of images loaded ({len(imgs)}).')
    return img_nums, imgs
# Define function to save points layers as csv
def save_pts(viewer, layer_name, save_dir_path, csv_name=None):
    layer = viewer.layers[layer_name]
    if not isinstance(layer, napari.layers.points.points.Points):
        raise ValueError(f'{layer_name} is not an active napari Points layer.')
    # Make sure save_dir_path is a Path object
    save_dir_path = Path(save_dir_path)
    if not save_dir_path.is_dir():
        save_dir_path.mkdir(parents=True)
    if csv_name is None:
        csv_name = layer.name
    csv_path = Path(f'{save_dir_path}/{csv_name}.csv')
    if csv_path.exists():
        print(f'{layer.name}.csv already exists.')
    else:
        layer.save(csv_path)
        print(f'{layer.name}.csv saved.')

