import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def save_GIF(array_list, output_file='animation.gif', interval=200, cmap_ = 'jet', vmin = None, vmax = None, orient = 'vertical'):
    """
    Create a GIF animation from a list of 2D NumPy arrays.

    Parameters:
    - array_list: list of 2D NumPy arrays representing frames.
    - output_file: string, filename for the output GIF (e.g., 'animation.gif').
    - interval: int, delay between frames in milliseconds.
    """

    fig, ax = plt.subplots()
    # Optional: Turn off the axis if you prefer
    # ax.axis('off')

    # Initialize the image with the first frame
    im = ax.imshow(array_list[0], animated=True, cmap=cmap_)
    # Create the colorbar
    cbar = fig.colorbar(im, ax=ax, orientation = orient)

    # Update function for animation
    def update(frame):
        im.set_array(frame)
        # Update color limits based on current frame's data
        im.set_clim(vmin=vmin, vmax=vmax)
        # Update the colorbar to match the new color limits
        cbar.update_normal(im)
        return [im]

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=array_list, interval=interval, blit=False
    )

    # Save the animation as a GIF
    ani.save(output_file, writer='pillow')

    plt.close(fig)  # Close the figure to prevent it from displaying

# Example usage:
# Suppose you have a list of 2D NumPy arrays called 'frames'
# frames = [array1, array2, array3, ...]
# create_gif_from_arrays(frames, output_file='my_animation.gif', interval=100)
