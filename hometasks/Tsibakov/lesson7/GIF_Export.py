import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import convolve
from matplotlib.colors import Normalize


def apply_cmap(data, cmap_name, vmin, vmax):
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    colormap = plt.get_cmap(cmap_name)

    normalized_data = norm(data)
    mapped_data = colormap(normalized_data)

    RGB_data = (mapped_data[:, :, :3] * 255).astype(np.uint8)
    
    return RGB_data


def save_GIF(data, scale, cmap, vmin, vmax, smooth, save_name = 'result.gif', duration_ms = 30):
    if (scale > 1):
        data = np.kron(data, np.ones((scale, scale)))

    if (smooth == True):
        data = [convolve(frame, np.ones((3,3))/9) for frame in data]

    RGB_data = [apply_cmap(frame, cmap, vmin, vmax) for frame in data]

    imgs = [Image.fromarray(img) for img in RGB_data]

    imgs[0].save(save_name, save_all = True, append_images = imgs[1:], duration = duration_ms, loop=0)
    