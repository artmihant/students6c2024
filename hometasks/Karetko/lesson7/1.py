#Задача об одноосно деформироованном состоянии 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# PHYSICS
lX = 10.0
K = 1.0
rho = 1.0
G = 0.5
# NUMERICS
nX = 100
nSteps = 100
cfl = 1.0

# PREPROCESSING
x = np.linspace(-0.5 * lX, 0.5 * lX, nX)
dX = lX / (nX - 1)
dt = cfl * dX / np.sqrt((K + 4* G / 3) / rho)

# INITIAL CONDITIONS
p0 = 1.0
p = np.exp(-p0 * x * x)
v = np.zeros(nX + 1)
tau = np.zeros(nX)


fig = plt.figure()
ax = plt.axes(xlim=(-0.5 * lX, 0.5 * lX), ylim=(-0.5, 1))
ax.set_xlabel("x")
ax.set_ylabel("p")
graph, = ax.plot([], [], lw=3)

def init():
    graph.set_data([], [])
    return graph,
# ACTION LOOP
def ActionLoop(i):

    global p, v, tau
    dpdt = (-np.diff(v))/ dX * K
    p = p + dpdt * dt
    dvdt = (-np.diff(p) + np.diff(tau)) / dX / rho
    v[1:-1] = v[1:-1] + dvdt * dt
    dtaudt = (np.diff(v))/ dX * G  * 4 / 3
    tau = tau + dtaudt * dt
   
    graph.set_data(x, p)
    ax.set_title(f"time step number: {i}")

    return graph,

anim = ani.FuncAnimation(fig=fig, func=ActionLoop,init_func=init, frames=nSteps,interval=20, blit=True)
anim.save("p.gif", writer=ani.PillowWriter(fps=30))