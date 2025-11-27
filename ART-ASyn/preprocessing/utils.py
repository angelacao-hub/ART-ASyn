import base64
import numpy as np

from PIL import Image
from torch import Tensor
from IPython.display import display, HTML

def display_images(images, space=10):
    html = f'<div style="display: flex; gap: {space}px; align-items: center;">'
    for img in images:
        if isinstance(img, Tensor):
            img = img.detach().cpu().numpy()
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img_str = base64.b64encode(img._repr_png_()).decode()
        html += f'<img src="data:image/png;base64,{img_str}" style="margin: 0; border: 1px solid black;">'
    html += '</div>'
    display(HTML(html))

import os
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def normalize(tensor, dim=(-1, -2), keepdim=False):
    if keepdim:
        min_val = tensor.amin(dim=dim, keepdim=True)
        max_val = tensor.amax(dim=dim, keepdim=True)
    else:
        min_val = tensor.min()
        max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val + 1e-10)


from PIL import Image
from torch import Tensor
import matplotlib.pyplot as plt
def get_cmap_image(array, cmap_name='gray'):
    """
    Apply a matplotlib colormap to a NumPy array and return a PIL image.
    
    :param array: 2D numpy array with float values in range [0, 1]
    :param cmap_name: name of the matplotlib colormap (e.g., 'plasma', 'viridis')
    :return: PIL.Image with RGB values
    """

    if isinstance(array, Tensor):
        array = array.detach().cpu().numpy()
    assert array.ndim == 2, "Input must be a 2D array"
    assert np.min(array) >= 0 and np.max(array) <= 1, "Values should be in [0, 1] range"

    # Get the colormap
    cmap = plt.get_cmap(cmap_name)

    # Apply the colormap (returns RGBA in [0, 1])
    colored_array = cmap(array)

    # Convert to 8-bit RGB image
    colored_array_uint8 = (colored_array[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored_array_uint8)