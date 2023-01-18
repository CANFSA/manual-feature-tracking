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
    ----------
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
    -------
    Returns
    -------
    tuple
        2-tuple containing np.ndarray of image numbers and N x Height x Width np.ndarray representing images
    ------
    Raises
    ------
    ValueError
        If img_dir directory does not exist.
    ValueError
        If img_dir directory has no images of file_suffix type.
    ValueError
        If len(manual_img_nums) doesn't match number of loaded images
    """
    img_dir = Path(img_dir_path)
    if not img_dir.is_dir():
        raise ValueError(
            f'Directory not found: '
            f'{img_dir}'
        )
    img_fns = [fn for fn in img_dir.glob(f'*{file_suffix}')]
    img_fns.sort()
    if len(img_fns) == 0:
        raise ValueError(
            f'No images of type "{file_suffix}" found in directory: '
            f'{img_dir}'
        )
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
    imgs = np.stack(imgs)
    return img_nums, imgs

def process_images(
    imgs,
    process='div_by_pre'
):
    """Process images by dividing by preceeding image, dividing by first image, subtracting preceeding image, or subtracting first image.

    Parameters
    ----------
    imgs : np.ndarray
        N x Height x Width np.ndarray representing images to be processed
    process : str, optional
        String corresponding to processing method; must be one of ['div_by_pre', 'div_by_first', 'sub_pre', 'sub_first'], by default 'div_by_pre'

    Returns
    -------
    np.ndarray
        N x Height x Width array representing processed images

    Raises
    ------
    ValueError
        If process not in ['div_by_pre', 'div_by_first', 'sub_pre', 'sub_first']
    """
    if process == 'sub_pre':
        processed_imgs = [
            imgs[n, :, :] - imgs[n - 1, :, :] for n in range(1, imgs.shape[0])
        ]
        processed_imgs.insert(0, np.zeros_like(processed_imgs[0]))
    elif process == 'sub_first':
        processed_imgs = [
            imgs[n, :, :] - imgs[0, :, :] for n in range(1, imgs.shape[0])
        ]
        processed_imgs.insert(0, np.zeros_like(processed_imgs[0]))
    elif process == 'div_by_pre':
        processed_imgs = [
            imgs[n, :, :] / imgs[n - 1, :, :] for n in range(1, imgs.shape[0])
        ]
        processed_imgs.insert(0, np.zeros_like(processed_imgs[0]))
    elif process == 'div_by_first':
        processed_imgs = [
            imgs[n, :, :] / imgs[0, :, :] for n in range(1, imgs.shape[0])
        ]
        processed_imgs.insert(0, np.zeros_like(processed_imgs[0]))
    else:
        raise ValueError(f"Processing routine {process} not recognized; process must be 'sub_pre', 'sub_first', 'div_by_pre', or 'div_by_first'.")
    processed_imgs = np.stack(processed_imgs)
    return processed_imgs

def save_points(viewer, layer_name, save_dir_path=None, csv_name=None):
    """Save napari points layer as a CSV.

    Parameters
    ----------
    viewer : napari.Viewer
        napari Viewer object where points layer exists
    layer_name : str
        Name of points layer as labeled in napari viewer
    save_dir_path : str, optional
        Path to directory where CSV will be saved; if None, will be saved in current working directory, by default None
    csv_name : str, optional
        Name for CSV file; if None, will use Points layer name with spaces replaced by hyphens, by default None

    Raises
    ------
    ValueError
        If layer_name is not a napari Points layer
    """
    layer = viewer.layers[layer_name]
    if not isinstance(layer, napari.layers.points.points.Points):
        raise ValueError(f'{layer_name} is not an active napari Points layer.')
    if save_dir_path is not None:
        save_dir_path = Path(save_dir_path)
    else:
        # Set save_dir_path as current working directory (location of the notebook)
        save_dir_path = Path.cwd()
    if not save_dir_path.is_dir():
        save_dir_path.mkdir(parents=True)
    if csv_name is None:
        csv_name = layer.name
        csv_name = csv_name.replace(' ', '-')
    csv_path = Path(f'{save_dir_path}/{csv_name}.csv')
    csv_path = csv_path.resolve()
    if csv_path.exists():
        print(f'File already exists: {csv_path}')
    else:
        layer.save(csv_path)
        print(f'CSV saved: {csv_path}')

def load_points(csv_path):
    """Load points data from a CSV file to be added to a napari viewer.

    Parameters
    ----------
    csv_path : str or Path
        Path to a CSV file containing Points layer data

    Returns
    -------
    tuple
        2-tuple containing CSV name and points data in an array of coordinates (slices, row, column) that can be read by napari.add_points()
    """
    csv_path = Path(csv_path)
    csv_name = csv_path.stem
    points_data = napari.utils.io.csv_to_layer_data(csv_path)[0]
    return csv_name, points_data

