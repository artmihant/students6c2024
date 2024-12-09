import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


lx = 10.0
k = 1.0
rho = 1.0
G = 0.5
nx = 100
x = np.linspace(-0.5 * lx, 0.5 * lx, nx)
dx = lx / (nx - 1)

nt = 1000
cfl = 1  
dt = cfl * dx / np.sqrt((k + 4.0 / 3.0 * G) / rho)


p0 = 1.0
p = p0 * np.exp(-x * x)
v = np.zeros(nx + 1)
tau = np.zeros(nx)


fig, ax = plt.subplots()
graph, = ax.plot(x, tau, label='tau')
ax.set_xlim(-0.5 * lx, 0.5 * lx)
ax.set_ylim(-1,1)
ax.set_title("tau(x,t)")
ax.set_xlabel("x")
ax.set_ylabel("tau")
ax.legend()

def update(frame):
    global p, v, tau
    

    dp = np.diff(p)
    dv = np.diff(v)
    

    p -= (k * dv / dx) * dt
    tau += (4.0 / 3.0 * G * dv / dx) * dt
    v[1:-1] += (-np.diff(p) + np.diff(tau)) / dx / rho * dt

    graph.set_ydata(tau)
    ax.set_title(f"Time Step: {frame}")
    
    return graph,


ani = FuncAnimation(fig, update, frames=nt, blit=True)


ani.save("tau(x,t).gif", writer=PillowWriter(fps=60))
print('saved')
#plt.show()
