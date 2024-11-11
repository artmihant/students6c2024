import time
import matplotlib.pyplot as plt
import numpy as np

width, height = 1920, 1080
max_iter = 128

x_min, x_max = -2.2, 1
y_min, y_max = (x_min-x_max)/2, (x_max-x_min)/2

z0 = complex(0, 0)


@np.vectorize
def calc_iter(z0,c):

    z = z0
    iter = 0

    for i in range(max_iter):
        if abs(z) > 2:
            iter = i
            break
        z = z * z + c
    else:
        iter = max_iter 

    return iter


def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, z0):
    divergence_step = np.zeros((height,width), dtype='int')

    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    real, imag = np.meshgrid(real, imag)
    c = real + 1j * imag
    
    divergence_step = calc_iter(z0,c)

    return divergence_step 

now = time.time()
image = mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, z0)
print(time.time() - now)

plt.imshow(image, cmap='hot', extent=(x_min, x_max, y_min, y_max))
plt.colorbar(label="Iters before devergence")
plt.title("Mandelbrot fractal")
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()

