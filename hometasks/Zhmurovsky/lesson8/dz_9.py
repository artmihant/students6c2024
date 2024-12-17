
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
p0 = 0.0
p = p0 * np.exp(-x**2 - y**2)
tauxx = np.zeros((nx, ny))
tauyy = np.zeros((nx, ny))
tauxy = np.zeros((nx - 1, ny - 1))
ux = np.zeros((nx + 1, ny))
uy = np.zeros((nx, ny + 1))
ux[0,:] = 0.01
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
    global p, tauxx, tauyy, tauxy, vx, vy, ux, uy
    div_u = np.diff(ux, 1, 0) / dx + np.diff(uy, 1, 1) / dy
    p = - div_u * K

    tauxx =  (np.diff(ux, 1, 0) / dx - div_u / 3.0) * 2.0 * G
    tauyy =  (np.diff(uy, 1, 1) / dy - div_u / 3.0) * 2.0 * G
    tauxy =  (np.diff(ux[1:-1, :], 1, 1) / dy + np.diff(uy[:, 1:-1], 1, 0) / dx) * G

    dvxdt = (np.diff(-p[:, 1:-1] + tauxx[:, 1:-1], 1, 0) / dx + np.diff(tauxy, 1, 1) / dy) / rho
    vx[1:-1, 1:-1] = (1 - dmp*dt) * vx[1:-1, 1:-1] + dvxdt * dt

    dvydt = (np.diff(-p[1:-1, :] + tauyy[1:-1, :], 1, 1) / dy + np.diff(tauxy, 1, 0) / dx) / rho
    vy[1:-1, 1:-1] = (1 - dmp*dt) * vy[1:-1, 1:-1] + dvydt * dt
    ux[1:-1, 1:-1] = ux[1:-1, 1:-1] + vx[1:-1, 1:-1] * dt
    uy[1:-1, 1:-1] = uy[1:-1, 1:-1] + vy[1:-1, 1:-1] * dt    

    fig.suptitle(str(i+1))
    graph.set_array(p.ravel())
    return graph,

ani = ani.FuncAnimation(fig, update_iter, frames=nsteps, interval=50, blit=True)
ani.save('pseudo-transient_deformatons.gif', writer='imagemagick')
plt.show()
