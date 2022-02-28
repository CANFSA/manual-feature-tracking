# Standard library imports
from pathlib import Path
# Third-party imports
import napari
import pandas as pd

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

