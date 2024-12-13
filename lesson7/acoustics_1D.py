import numpy as np
import matplotlib.pyplot as plt

# PHYSICS
lX = 10.0
K = 1.0
rho = 1.0

# NUMERICS
nX = 100
nSteps = 100
cfl = 1.0

# PREPROCESSING
x = np.linspace(-0.5 * lX, 0.5 * lX, nX)
dX = lX / (nX - 1)
dt = cfl * dX / np.sqrt(K / rho)

# INITIAL CONDITIONS
p0 = 1.0
p = np.exp(-p0 * x * x)
v = np.zeros(nX + 1)

plt.ion()    # interactive mode
graph = plt.plot(x, p)[0]
plt.pause(0.1)

# ACTION LOOP
for i in range(nSteps):
    dpdt = -np.diff(v) / dX * K
    p = p + dpdt * dt
    dvdt = -np.diff(p) / dX / rho
    v[1:-1] = v[1:-1] + dvdt * dt
    graph.remove()
    graph = plt.plot(x, p)[0]
    plt.title(str(i+1))
    plt.pause(0.001)
