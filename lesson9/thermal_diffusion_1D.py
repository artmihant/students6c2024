import numpy as np
import matplotlib.pyplot as plt

# PHYSICS
lx = 10.0
lam = 1.0
rhocp = 1.0

# NUMERIC
nx = 100
nsteps = 1000
cfl = 1.0

# PREPROCESSING
x = np.linspace(-0.5 * lx, 0.5 * lx, nx)
dx = lx / (nx - 1)
dt = cfl * dx * dx / lam / rhocp / 2.0

# INITIAL CONDITIONS
T0 = 1.0
T = T0 * np.exp(-x * x)
q = np.zeros(nx - 1)

plt.ion()    # interactive mode
graph = plt.plot(x, T)[0]

# ACTION LOOP
for i in range(nsteps):
    q = -lam * np.diff(T) / dx
    dTdt = -np.diff(q) / dx / rhocp
    T[1:-1] = T[1:-1] + dTdt * dt
    graph.remove()
    graph = plt.plot(x, T)[0]
    plt.title(str(i+1))
    plt.pause(0.0001)
