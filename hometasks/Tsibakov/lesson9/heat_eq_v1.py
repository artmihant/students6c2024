"""
 Tsybakov Mikhail 626 
 15.12.2024
"""



import numpy as np
import matplotlib.pyplot as plt



L = 1.0       # Domain size
nx = 100  
dx = L / (nx - 1) 

alpha = 0.01  # Thermal diffusivity
x0 = L / 2    # Point source location

t1 = 1.0
nt = 1000  
dt = t1 / nt  



# CFL check
cfl = alpha * dt / dx**2
if cfl >= 0.5:
    raise ValueError("CFL condition not satisfied: alpha * dt / dx^2 < 0.5")



# Initial conditions
x = np.linspace(0, L, nx)
u = np.zeros(nx)

u[np.argmin(np.abs(x - x0))] = 1.0 / dx  # Delta-like source with unit energy



# Explicit Euler time integration
u_new = np.zeros_like(u)
for n in range(nt):

    for i in range(1, nx - 1):
        u_new[i] = u[i] + alpha * dt / dx**2 * (u[i - 1] - 2 * u[i] + u[i + 1])

    u[:] = u_new[:]



# Comparing with analytics
def heat_equation_point_source(x, t, x0, alpha):
    return (1 / np.sqrt(4 * np.pi * alpha * t)) * np.exp(-((x - x0)**2) / (4 * alpha * t))

u_analyt = heat_equation_point_source(x, t1, x0, alpha)



plt.figure(figsize=(8, 6))
plt.plot(x, u, label = "Numerical solution", lw = 3, alpha = 0.2, color = "blue")
plt.plot(x, u_analyt, label = "Analytic solution: Gaussian distribution", lw = 0.75, color = "black")
plt.title("1D Heat Equation with Explicit Time Integration", font = "Consolas", fontsize = 10)
plt.xlabel("x", font = "Consolas", fontsize = 10)
plt.ylabel("u(x, t)", font = "Consolas", fontsize = 10)
plt.legend()
plt.grid(color = "black", alpha = 0.15)
plt.savefig("compare_1.png", dpi = 300)
plt.show()