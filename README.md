# manual-feature-tracking

Manually track features across a sequence of images using the Python package napari. Repo contains a Python Jupyter notebook that interfaces with a napari viewer window to allow a user to place points annotating a feature. 3D coordinates of points (slice, row, column) can be exported as a CSV file.

## Requirements

Package requirements are listed in [requirements.txt](requirements.txt) as minimum requirements. This projects depends heavily on the GUI that [napari](https://napari.org/) offers, so this dependency is important, and requires Python 3.7, 3.8, or 3.9. The project was developed with Python 3.9.5.

## Installation

All packages used besides napari are included in the Anaconda distribution of Python, so that is the easiest way to get started. To install napari from conda:

```
conda install -c conda-forge napari
```
After installing with conda, napari should be updated with the following command:
```
conda update napari
```

To install napari using pip (if you aren't using Anaconda):

```
pip install "napari[all]"
```

Further napari installation information can be found [here](https://napari.org/tutorials/fundamentals/installation.html).

## Running the project

This project exists as a series of Jupyter notebooks which execute helper functions sotred in the [helpers.py](helpers.py) module.

### [manual-tracking-and-saving.ipynb](manual-tracking-and-saving.ipynb)
This notebook can be used to open a sequence of images in napari (with or without specifying starting image, stopping image, and step between images) and track features by placing points in three dimensions (slice, row, column). Once the points are added to napari, they can be saved to a CSV file.

### [load-and-edit-points.ipynb](load-and-edit-points.ipynb)

This notebook is used to load a CSV containing coordinates of points (slice, row, column) and add them to napari as a Points layer. This will allow the points to be edited and saved again to a CSV file.
