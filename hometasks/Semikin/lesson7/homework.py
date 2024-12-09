import numpy as np
import matplotlib.pyplot as plt

lx = 10.0
k = 1.0
rho = 1.0
nx = 100
nsteps = 1000
cfl = 1
g = 0.5

x = np.linspace(-0.5*lx,0.5*lx,nx)
dx=lx/(nx-1)
dt=cfl*dx/np.sqrt((k + 4.0/3.0 * g)/rho)

p0 = 1.0
p = p0 * np.exp(-x*x)
v = np.zeros(nx+1)
tau = np.zeros(nx)

plt.ion()
plt.title('0')
graph = plt.plot(x,p)[0]
#graph1 = plt.plot(x,tau)[0]
plt.pause(0.0001)
for i in range(nsteps):
    #p_old = p
    #tau_old = tau
    p = p - np.diff(v)/dx*k*dt
    tau = tau + np.diff(v)/dx * 4.0/3.0 * g * dt
    v[1:-1] = v[1:-1] + (np.diff(tau)-np.diff(p))/(dx*rho)*dt
    graph.remove()
    #graph1.remove()
    graph = plt.plot(x,p)[0]
    #graph1 = plt.plot(x,tau)[0]
    plt.title(str(i+1))
    plt.pause(0.0001)
plt.ioff()
plt.show()
