
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Physics
lx = 10.0
ly = 10.0
K = 1.0
G = 0.5
rho = 1.0

# Numerics
nx = 200
ny = 200
nsteps = 400
cfl = 0.5
dmp = 4.0 / nx

# Preprocessing
x = np.linspace(-0.5 * lx, 0.5 * lx, nx)
y = np.linspace(-0.5 * ly, 0.5 * ly, ny)
x, y = np.meshgrid(x, y, indexing='ij')
dx = lx / (nx - 1)
dy = ly / (ny - 1)
dt = cfl * min(dx, dy) / np.sqrt((K + 4.0 * G / 3.0) / rho)

# Initial conditions
p0 = 1.0
p = p0 * np.exp(-x**2 - y**2)
tauxx = np.zeros((nx, ny))
tauyy = np.zeros((nx, ny))
tauxy = np.zeros((nx - 1, ny - 1))
vx = np.zeros((nx + 1, ny))
vy = np.zeros((nx, ny + 1))

# plot
fig, ax = plt.subplots()
graph = ax.pcolormesh(x, y, p, shading='auto')
ax.axis('scaled')
ax.set_title('P')
fig.colorbar(graph, ax=ax)

# iteration
def update_iter(i):
    global p, tauxx, tauyy, tauxy, vx, vy
    div_v = np.diff(vx, 1, 0) / dx + np.diff(vy, 1, 1) / dy
    p = p - div_v * K * dt

    tauxx = tauxx + (np.diff(vx, 1, 0) / dx - div_v / 3.0) * 2.0 * G * dt
    tauyy = tauyy + (np.diff(vy, 1, 1) / dy - div_v / 3.0) * 2.0 * G * dt
    tauxy = tauxy + (np.diff(vx[1:-1, :], 1, 1) / dy + np.diff(vy[:, 1:-1], 1, 0) / dx) * G * dt

    dvxdt = (np.diff(-p[:, 1:-1] + tauxx[:, 1:-1], 1, 0) / dx + np.diff(tauxy, 1, 1) / dy) / rho
    vx[1:-1, 1:-1] = (1 - dmp*dt) * vx[1:-1, 1:-1] + dvxdt * dt

    dvydt = (np.diff(-p[1:-1, :] + tauyy[1:-1, :], 1, 1) / dy + np.diff(tauxy, 1, 0) / dx) / rho
    vy[1:-1, 1:-1] = (1 - dmp*dt) * vy[1:-1, 1:-1] + dvydt * dt

    fig.suptitle(str(i+1))
    graph.set_array(p.ravel())
    return graph,

ani = ani.FuncAnimation(fig, update_iter, frames=nsteps, interval=50, blit=True)
ani.save('pseudo-transient.gif', writer='imagemagick')
plt.show()


