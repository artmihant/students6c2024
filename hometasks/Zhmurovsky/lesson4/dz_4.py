import numpy as np
import matplotlib.pyplot as plt

width, height = 800, 800
max_iter = 200

x_min, x_max = -2, 2
y_min, y_max = (x_min-x_max)/2, (x_max-x_min)/2

z0 = complex(-1, 0)

def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, z0):
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    real, imag = np.meshgrid(real, imag)
    c = real + 1j * imag

    z = np.full(c.shape, z0, dtype=np.complex128)
    divergence_step = np.full(c.shape, max_iter, dtype=int)

    for i in range(max_iter):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] ** 2 + c[mask]

        diverged = (np.abs(z) > 2) & (divergence_step == max_iter)
        divergence_step[diverged] = i

    return divergence_step

image = mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, z0)

plt.imshow(image, cmap='hot', extent=(x_min, x_max, y_min, y_max))
plt.colorbar(label="Iters before devergence")
plt.title("Mandelbrot fractal")
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()

