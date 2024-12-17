import numpy as np
import matplotlib.pyplot as plt

lx = 10.0
ly = 10.0
k = 1.0
rho = 1.0
nx = 100
ny = 100
nsteps = 1000
cfl = 0.5
g = 0.5
dmp = 0

x = np.linspace(-0.5*lx,0.5*lx,nx)
y = np.linspace(-0.5*ly,0.5*ly,ny)
x, y = np.meshgrid(x, y, indexing='ij')
dx=lx/(nx-1)
dy=ly/(ny-1)
dt=cfl*min(dx,dy)/np.sqrt((k + 4.0/3.0 * g)/rho)

p0 = 1.0
p = p0 * np.exp(-x*x - y*y)
tauxx = np.zeros((nx,ny))
tauyy = np.zeros((nx,ny))
tauxy = np.zeros((nx-1,ny-1))
vx = np.zeros((nx+1,ny))
vy = np.zeros((nx,ny+1))

plt.ion()
plt.title('0')
graph = plt.pcolormesh(x, y, p)
plt.colorbar()
#graph1 = plt.plot(x,tau)[0]
plt.pause(0.0001)
for i in range(nsteps):
    #p_old = p
    #tau_old = tau
    div_v = np.diff(vx,1,0) / dx + np.diff(vy,1,1) / dy
    p = p - div_v*k*dt
    tauxx = tauxx + (np.diff(vx,1,0) / dx - div_v / 3.0) * 2.0 * g * dt
    tauyy = tauyy + (np.diff(vy,1,1) / dy - div_v / 3.0) * 2.0 * g * dt
    tauxy = tauxy + (np.diff(vx[1:-1, :],1,1) / dy + np.diff(vy[:, 1:-1],1,0) / dx) * g * dt
    dvxdt = np.diff(-p[:,1:-1] + tauxx[:,1:-1],1,0) / dx + np.diff(tauxy,1,1) / dy
    vx[1:-1, 1:-1] = (1 - dmp) * vx[1:-1, 1:-1] + dvxdt/rho*dt
    dvydt = np.diff(-p[1:-1,:] + tauyy[1:-1,:],1,1) / dy + np.diff(tauxy,1,0) / dx
    vy[1:-1, 1:-1] = (1 - dmp) * vy[1:-1, 1:-1] + dvydt/rho*dt
    graph.remove()
    #graph1.remove()
    graph = plt.pcolormesh(x,y,p)
    #graph1 = plt.plot(x,tau)[0]
    plt.title(str(i+1))
    plt.pause(0.0001)
plt.ioff()
plt.show()
