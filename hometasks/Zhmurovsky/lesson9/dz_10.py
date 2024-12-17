
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Physics
lx = 10.0
lam = 1.0
rhocp = 1.0

# Numerics
nx = 100
nt = 100
cfl = 1.0

# Preprocessing
x = np.linspace(0, lx, nx)
dx = lx / (nx - 1)
dt = cfl * dx**2 * rhocp / lam / 2.0

# Initial Conditions
T0 = 1.0
T = T0 * np.sin(np.pi * x / lx)
q = np.zeros(nx - 1)
t = 0.0

# Analytical solution function
def analytical_solution(x, t, T0):
     return np.sin(np.pi * x / lx) * np.exp( - (lam / rhocp) * (np.pi/lx)**2 * t)

# Plot setup
fig, ax = plt.subplots()
ax.set_xlim(0 , lx)
ax.set_ylim(0, T0)
line, = ax.plot(x, T, label="Numerical Solution", color="blue")
analytical_line, = ax.plot(x, T - analytical_solution(x, 0, T0), label="Analytical Solution", linestyle="--", color="red")
step_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

ax.set_xlabel("x")
ax.set_ylabel("T")
ax.legend()

# Update function for animation
def update_iter(i):
    global T, t
    q = -lam * np.diff(T) / dx
    T[1:-1] = T[1:-1] - np.diff(q) / dx / rhocp * dt
    line.set_ydata(T)
    t = t + dt
    analytical_line.set_ydata(analytical_solution(x, t, T0))
    step_text.set_text(f"Step: {i}")
    return line, analytical_line, step_text


animation = ani.FuncAnimation(fig, update_iter, frames=nt, interval=20, blit=True)
animation.save('thermal_conductivity.gif', writer='imagemagick')
plt.show()


