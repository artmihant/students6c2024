import time
import numpy as np
import matplotlib.pyplot as plt

width, height = 800, 800
max_iter = 32

x_min, x_max = -2, 1
y_min, y_max = (x_min-x_max)/2, (x_max-x_min)/2

z0 = complex(0, 0)

def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, z0):
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    real, imag = np.meshgrid(real, imag)
    c = real + 1j * imag

    z = np.full(c.shape, z0, dtype=np.complex128)
    divergence_step = np.full(c.shape, max_iter, dtype=int)

    mask = divergence_step == max_iter

    for i in range(max_iter):

        z[mask] = z[mask] ** 2 + c[mask]

        divergence_step[mask & (np.abs(z) > 2)] = i
        mask = mask & (np.abs(z) < 2)

    divergence_step = np.log2(divergence_step+1)

    return divergence_step

# for max_iter in range(128):

now = time.time()

image = mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, z0)

print(time.time() - now)

fig, ax = plt.subplots() 

ax.imshow(image, cmap='afmhot', extent=(x_min, x_max, y_min, y_max))
# fig.savefig(f'{max_iter}.png')
# plt.close(fig)

plt.show()

