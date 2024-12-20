import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import random

X_min, X_max = -2,2
X_step = 1000

Y_min, Y_max = -2,2
Y_step = 1000

z0 = complex(0.0, 0.0)
MaxIter = 60

XSpace = np.linspace(X_min, X_max, X_step).reshape((1,-1))
YSpace = np.linspace(Y_min, Y_max, Y_step).reshape((-1,1))

def mandelbrot(real, imag, max_iter, z0):
    c = real + 1j * imag
    z = np.full(c.shape, z0, dtype=np.complex128)
    divergence_step = np.full(c.shape, max_iter, dtype=int)

    for i in range(max_iter):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] ** 2 + c[mask]

        diverged = (np.abs(z) > 2) & (divergence_step == max_iter)
        divergence_step[diverged] = i

    return divergence_step

for i in range(5):
  image =  mandelbrot(XSpace,YSpace, MaxIter, z0)
  Fig, Ax = plt.subplots()
  x_grid, y_grid = np.meshgrid(np.linspace(X_min,X_max,X_step), np.linspace(Y_min,Y_max,Y_step))
  graph = Ax.pcolormesh(x_grid,y_grid,image,cmap='inferno')
  Ax.set_xlabel("Re")
  Ax.set_ylabel("Im")
  Ax.set_title(f"z: {z0}")
  plt.show()
  z0 = complex(random.uniform(-1, 1),random.uniform(-1, 1))