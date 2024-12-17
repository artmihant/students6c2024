import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm
import pyvista as pv

Viscosity = 2.0                           # вязкость жидкости
n_x, n_y, n_z = 30, 30, 100                # размеры решетки
num_iterations = 100
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
radius = 10
barrier = np.zeros((n_x, n_y, n_z), dtype=bool)
for x in range(n_x):
    for y in range(n_y):
        for z in range(n_z):
            if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 < radius**2:
                barrier[x, y, z] = True

for i in range(19):
    eu = np.sum(V[i] * u, axis=-1)
    f[..., i] = W[i] * rho * (1 + eu / c_s**2 + (eu**2) / (2 * c_s**4) - np.sum(u**2, axis=-1) / (2 * c_s**2))

plots = np.zeros(shape = (num_iterations, n_x, n_z))

for step in tqdm(range(num_iterations)):
    # вычисление скоростей и плотности
    rho = np.sum(f, axis=-1)
    
    plots[step] = rho[:, n_y // 2 - 1, :]  

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
        f_new[barrier, i] = f[barrier, opposite[i]]
    
    f = f_new
    
fig, ax = plt.subplots()
cax = ax.imshow(plots[0], origin="lower", cmap="viridis")
fig.colorbar(cax)

def update(frame):
    cax.set_array(plots[frame])
    ax.set_title(f"Time step: {frame}")  
    return [cax]

anim = FuncAnimation(fig, update, frames=num_iterations - 1, blit=True, interval=100)
anim.save("rho.gif", writer="pillow", fps=20)