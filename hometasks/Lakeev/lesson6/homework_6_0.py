import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pyvista as pv

Viscosity = 10.0                           # вязкость жидкости
n_x, n_y, n_z = 30, 30, 100          # размеры решетки
num_iterations = 1000
c_s = 1 / np.sqrt(3)                       # скорость звука
U0 = np.array([0.0, 0.0, 0.10])      

D = 3 # Мерность модели
Q = 19 # число точек в шаблоне

V = np.array([
    [0, 0, 0],
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
    [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
    [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]
])

#противоположное направление для каждого вектора
opposite = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]

W = np.array([
    1 / 3,
    1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
    1 / 36, 1 / 36, 1 / 36, 1 / 36,
    1 / 36, 1 / 36, 1 / 36, 1 / 36,
    1 / 36, 1 / 36, 1 / 36, 1 / 36
])

#инициализация полей
rho = np.ones((n_x, n_y, n_z))  
u = np.zeros((n_x, n_y, n_z, 3)) 
u += U0
f = np.zeros((n_x, n_y, n_z, 19))  
feq = np.zeros((n_x, n_y, n_z, 19))  

# Сферический барьер
cx, cy, cz = (n_x - 1) / 2 , (n_y - 1) / 2, (n_z - 1) / 2 
radius = 5
barrier = np.zeros((n_x, n_y, n_z), dtype=bool)
for x in range(n_x):
    for y in range(n_y):
        for z in range(n_z):
            if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 < radius**2:
                barrier[x, y, z] = True

for i in range(19):
    eu = np.sum(V[i] * u, axis=-1)
    f[..., i] = W[i] * rho * (1 + eu / c_s**2 + (eu**2) / (2 * c_s**4) - np.sum(u**2, axis=-1) / (2 * c_s**2))

for step in tqdm(range(num_iterations)):
    # вычисление скоростей и плотности
    rho = np.sum(f, axis=-1)
    u = np.zeros_like(u)  
    for i in range(19):
        u += f[..., i, np.newaxis] * V[i]  
    u /= rho[..., np.newaxis]  

    for i in range(19):
        eu = np.sum(V[i] * u, axis=-1)
        feq[..., i] = W[i] * rho * (1 + eu / c_s**2 + (eu**2) / (2 * c_s**4) - np.sum(u**2, axis=-1) / (2 * c_s**2))

    # Collide
    f = f - (1 / Viscosity) * (f - feq)

    # Stream
    f_new = np.zeros_like(f)

    for i in range(19):
        f_new[..., i] = np.roll(f[..., i], shift=V[i], axis=(0, 1, 2))

    f = f_new

    # условие рефлекта - частицы на границе идут в противоположную сторону 
    for i in range(19):
        f[barrier, i] = f[barrier, opposite[i]]  

x, y, z = np.meshgrid(
np.arange(n_x, dtype = np.float32),  
np.arange(n_y, dtype = np.float32),
np.arange(n_z, dtype = np.float32),
indexing='ij'
)   

factor = 6
x_sub = x[::factor, ::factor, ::factor]
y_sub = y[::factor, ::factor, ::factor]
z_sub = z[::factor, ::factor, ::factor]
rho_sub = rho[::factor, ::factor, ::factor]

sphere = pv.Sphere(radius=radius, center=(cx, cy, cz))
grid = pv.StructuredGrid(x_sub, y_sub, z_sub)
grid.point_data["density"] = rho_sub.flatten(order="F")

plotter = pv.Plotter()
plotter.add_volume(grid, scalars="density", cmap="magma", opacity="sigmoid")
plotter.add_mesh(sphere, color='skyblue', show_edges=True)
plotter.show()
