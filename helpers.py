# Standard library imports
from pathlib import Path
# Third-party imports
import imageio.v3 as iio
import napari
import numpy as np
import pandas as pd
from skimage import exposure, filters, util


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
        Index of image within directory which will be the first image loaded,
        by default None
    stop : int, optional
        Index of image within directory which will be the last image loaded,
        by default None
    step : int, optional
        Step size for images that are loaded between start and stop
        (e.g. 2 would be every other image), by default None
    manual_img_nums : array-like, optional
        List or array of integers that will be returned as the actual image
        numbers (only used if image directory is already a subset of the
        images), by default None
    file_suffix : str, optional
        File suffix for images located in img_dir_path, by default '.tif'
    convert_to_float : bool, optional
        If True, convert images to floating point images, by default True
    -------
    Returns
    -------
    np.ndarray, np.ndarray
        2-tuple containing np.ndarray of image numbers and
        N x Height x Width np.ndarray representing images
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
    # Image range is partially inclusive, so starting point has to be + 1
    # to include last image
    img_nums = np.arange(start, stop + 1, step)
    imgs = [iio.imread(img_fns[n]) for n in img_nums]
    if convert_to_float:
        imgs = [util.img_as_float(img) for img in imgs]
    if manual_img_nums is not None:
        img_nums = np.array(manual_img_nums)
    if len(imgs) != len(img_nums):
        raise ValueError(
            f'Length of manual_img_nums ({len(manual_img_nums)}) '
            f'must match number of images loaded ({len(imgs)}).'
        )
    imgs = np.stack(imgs)
    return img_nums, imgs

def process_images(imgs, method='div_by_pre', file_type='.tif'):
    """Process images by either dividing by previous image, subtracting
    previous image, dividing first image, or subtracting first image.
    ----------
    Parameters
    ----------
    imgs : numpy.ndarray
        3D NumPy array representing stack of 2D images (N x Height x Width)
    method : str, optional
        Processing method. One of:
         - 'div_by_pre' : Divide by previous image (default)
         - 'sub_pre' : Subtract previous image
         - 'div_by_first' : Divide first image
         - 'sub_first' : Subtract first image
    file_type : str, optional
        File suffix of images in directory, by default '.tif'
    -------
    Returns
    -------
    np.ndarray
        3D array of images with dimensions (images, rows, cols)
    ------
    Raises
    ------
    ValueError
        Raises ValueError if method not in list above
    """
    # Convert to float before smoothing
    imgs = util.img_as_float32(imgs)
    print('Smoothing images...')
    imgs = filters.gaussian(imgs)
    imgs_proc = np.zeros_like(imgs)
    # --------------------- #
    # Highlight differences #
    # --------------------- #
    if method == 'div_by_pre':
        print('Processing images by dividing previous...')
        imgs_proc[1:, :, :] = [
            imgs[i, :, :] / imgs[i - 1, :, :]
            for i in range(1, imgs.shape[0])
        ]
    elif method == 'sub_pre':
        print('Processing images by subtracting previous...')
        imgs_proc[1:, :, :] = [
            imgs[i, :, :] - imgs[i - 1, :, :]
            for i in range(1, imgs.shape[0])
        ]
    elif method == 'div_by_first':
        print('Processing images by dividing first...')
        imgs_proc[1:, :, :] = [
            imgs[i, :, :] / imgs[0, :, :]
            for i in range(1, imgs.shape[0])
        ]
    elif method == 'sub_first':
        print('Processing images by subtracting first...')
        imgs_proc[1:, :, :] = [
            imgs[i, :, :] - imgs[0, :, :]
            for i in range(1, imgs.shape[0])
        ]
    else:
        raise ValueError(
            'method must be "div_by_pre", "sub_pre", "div_by_first",'
            'or "sub_first"'
        )
    imgs_proc = np.stack(imgs_proc)
    return imgs_proc

def rescale_contrast(
    imgs,
    low=2.0,
    high=98.0,
    convert_to_16bit=True,
    convert_to_8bit=False,
):
    """Adjust contrast by rescaling and clipping image intensities.
    ----------
    Parameters
    ----------
    imgs : numpy.ndarray
        3D NumPy array representing stack of 2D images (N x Height x Width)
    low : float, optional
        Low percentile to which image will be rescaled, by default 2.0
    high : float, optional
        High percentile to which image will be rescaled, by default 98.0
    convert_to_16bit : bool, optional
        If True, convert to 16-bit image. Defaults to True.
    convert_to_8bit : bool, optional
        If True, convert to 8-bit image. Good for saving GIFs.
        Defaults to False.
    -------
    Returns
    -------
    numpy.ndarray
        3D NumPy array representing stack of rescaled 2D images
    """
    # --------------- #
    # Adjust contrast #
    # --------------- #
    print('Adjusting contrast...')
    # Determine intensity values for low & high percentiles
    p_low, p_high = np.percentile(imgs, [low, high])
    # Clip intensities less/greater than these intensities, respectively
    imgs_adj = np.clip(imgs, p_low, p_high)
    # Rescale float to [0, 1] so image can be converted to 16-bit
    imgs_adj = exposure.rescale_intensity(
        imgs_adj, out_range=(0, 1)
    )
    if convert_to_16bit:
        print('Converting to 16-bit images...')
        # Convert image to 16-bit
        imgs_adj = util.img_as_uint(imgs_adj)
    elif convert_to_8bit:
        print('Converting to 8-bit images...')
        # Convert image to 8-bit (good for saving GIFs)
        imgs_adj = util.img_as_ubyte(imgs_adj)
    return imgs_adj

def save_images(
    save_dir_path,
    imgs,
    start=0,
    stop=None,
    step=1,
    file_suffix='.tif',
    num_offset=0
):
    """Save 3D array of images as directory of 2D images, especially
    after processing.
    ----------
    Parameters
    ----------
    save_dir_path : str or Path
        Path to directory where images will be saved. Directory must not
        already exist, as a precaution for overwriting data accidentally.
    imgs : np.ndarray
        3xMxN NumPy array representing images along axis 0, rows along axis 1,
        columns along axis 2
    start : int, optional
        Index of imgs where images will begin for saving, by default 0
    stop : int or None, optional
        Index of imgs where images will stop for saving. Passing None will
        stop saving at end of array. Defaults to None.
    step : int, optional
        Step size between saved images, by default 1 (every image)
    file_suffix : str, optional
        Image type as which images will be saved. Defaults to '.tif'
    num_offset : int, optional
        Additional offset for image index when saving. Iterating index will
        be added to start to become name of saved images.
    ------
    Raises
    ------
    ValueError
        Raises ValueError if save_dir_path already exists
    """
    save_dir_path = Path(save_dir_path)
    if save_dir_path.exists():
        raise ValueError(
            'Directory already exists. Delete directory or change '
            'save_dir_path'
        )
    else:
        save_dir_path.mkdir(parents=True)
    if stop is None:
        stop = imgs.shape[0]
    if not file_suffix.startswith('.'):
        file_suffix = f'.{file_suffix}'
    print('Saving images...')
    n_imgs = 0
    for i in range(start, stop, step):
        img = imgs[i, :, :]
        # Save image with number adjusted by start offset and leading zeros
        # corresponding to length of stopping image number
        save_path = save_dir_path / (
            f'{str(i + num_offset).zfill(len(str(stop)))}{file_suffix}'
        )
        iio.imwrite(save_path, img)
        n_imgs += 1
    print(f'{n_imgs} image(s) saved to:')
    print(save_dir_path.absolute())

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

