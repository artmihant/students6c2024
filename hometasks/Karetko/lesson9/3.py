import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
#0.0002471210504259025
#phis
lx = 2*np.pi
lam = 1.0
rhocp = 1.0

#num
nx = 100
nt = 1000
cfl = 1.0

#preprocessing
x = np.linspace(-np.pi,np.pi,nx)
dx = lx/(nx-1)
dt =  cfl *dx**2*rhocp/lam/2.0
t=0
#init
T0 = 1.0
T = T0*np.sin(x)
q = np.zeros(nx - 1)


fig = plt.figure()
ax = plt.axes(xlim=(-0.5 * lx, 0.5 * lx), ylim=(-1, 1))
ax.set_xlabel("x")
ax.set_ylabel("T")
graph, = ax.plot([], [], lw=3)
t=0
T_analit = T
deltamax=0
def init():
    graph.set_data([], [])
    return graph,
# ACTION LOOP
def ActionLoop(i):
    global T_analit,t,x, deltamax
    #plt.plot(x, T, color='r', label='num')
    #plt.plot(x, T_analit, color='g', label='analit')
    #plt.ylim([-1,1])
    #plt.xlabel("x")
    #plt.ylabel("T")
    #plt.title(f"time step number: {i}")
    #plt.legend()
    #plt.show()
    q = - lam*np.diff(T)/dx
    T[1:-1] = T[1:-1] - np.diff(q)/dx/rhocp*dt
    graph.set_data(x, T)
    graph.set_data(x, T_analit)
    ax.set_title(f"time step number: {i}")
    t=t+dt
    a = np.sqrt(lam/rhocp)
    
    T_analit = a*np.exp(-t)*np.sin(x)
     
    delta = max(abs(T-T_analit))
    deltamax = max(delta,deltamax)
    print(deltamax)
    return graph,



anim = ani.FuncAnimation(fig=fig, func=ActionLoop,init_func=init, frames=nt,interval=20, blit=True)

anim.save("T2.gif", writer=ani.PillowWriter(fps=30))