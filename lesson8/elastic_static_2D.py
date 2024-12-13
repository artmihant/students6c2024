import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

def make_plot(fig, plot, x, y, field, name):
    gr = plot.pcolormesh(x, y, field, shading='auto')
    plot.axis('scaled')
    plot.set_title(name)
    fig.colorbar(gr, location='right')
    return gr

# PHYSICS
lX = 10.0
lY = 10.0
K = 1.0
G = 0.5
rho = 1.0

# NUMERICS
nX = 200
nY = 200
nSteps = 500
cfl = 0.5
dmp = 4.0 / nX

# PREPROCESSING
x = np.linspace(-0.5 * lX, 0.5 * lX, nX)
y = np.linspace(-0.5 * lY, 0.5 * lY, nY)
x, y = np.meshgrid(x, y, indexing='ij')    # 1D arrays x and y became 2D
dX = lX / (nX - 1)
dY = lY / (nY - 1)
dt = cfl * min(dX, dY) / np.sqrt((K + 4.0 * G / 3.0) / rho)

# INITIAL CONDITIONS
p0 = 1.0
p = p0 * np.exp(-x * x - y * y)
tauXX = np.zeros((nX, nY))
tauYY = np.zeros((nX, nY))
tauXY = np.zeros((nX - 1, nY - 1))
vX = np.zeros((nX + 1, nY))
vY = np.zeros((nX, nY + 1))

fig, graph = plt.subplots(2, 2)
gr = []
gr.append(make_plot(fig, graph[0, 0], x, y, p, 'p'))
gr.append(make_plot(fig, graph[0, 1], x, y, tauXX, 'tau_xx'))
gr.append(make_plot(fig, graph[1, 0], x, y, tauXX, 'tau_yy'))
gr.append(make_plot(fig, graph[1, 1], x[:-1, :-1], y[:-1, :-1], tauXY, 'tau_xy'))

# ACTION LOOP
def action_loop(i):
    divV = np.diff(vX, 1, 0) / dX + np.diff(vY, 1, 1) / dY
    p[:] = p - divV * K * dt
    tauXX[:] = tauXX + (np.diff(vX, 1, 0) / dX - divV / 3.0) * 2.0 * G * dt
    tauYY[:] = tauYY + (np.diff(vY, 1, 1) / dY - divV / 3.0) * 2.0 * G * dt
    tauXY[:] = tauXY + (np.diff(vX[1:-1, :], 1, 1) / dY + np.diff(vY[:, 1:-1], 1, 0) / dY) * G * dt
    dvXdt = (np.diff(-p[:, 1:-1] + tauXX[:, 1:-1], 1, 0) / dX + np.diff(tauXY, 1, 1) / dY) / rho
    vX[1:-1, 1:-1] = (1 - dmp) * vX[1:-1, 1:-1] + dvXdt * dt
    dvYdt = (np.diff(-p[1:-1, :] + tauYY[1:-1, :], 1, 1) / dY + np.diff(tauXY, 1, 0) / dX ) / rho
    vY[1:-1, 1:-1] = (1 - dmp) * vY[1:-1, 1:-1] + dvYdt * dt

    fig.suptitle(str(i+1))
    gr[0].set_array(p)
    gr[1].set_array(tauXX)
    gr[2].set_array(tauYY)
    gr[3].set_array(tauXY)
    gr[0].set_clim([p.min(), p.max()])
    gr[1].set_clim([tauXX.min(), tauXX.max()])
    gr[2].set_clim([tauYY.min(), tauYY.max()])
    gr[3].set_clim([tauXY.min(), tauXY.max()])
    return gr

anim = ani.FuncAnimation(fig=fig, func=action_loop, frames=nSteps, interval=10, repeat=False, repeat_delay=0)
plt.show()
