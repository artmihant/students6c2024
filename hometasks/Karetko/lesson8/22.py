import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
#метод установления в задаче с заданным начальным распределением давления

def make_plot(fig, plot, x, y, field, name):
    gr = plot.pcolormesh(x, y, field, shading='auto')
    plot.axis('scaled')
    plot.set_title(name)
    fig.colorbar(gr, location='right')
    return gr


lx = 10.0
ly = 10.0
k = 1.0
G = 0.25
rho = 1.0
nx = 200
ny = 200
nsteps = 1000
cfl = 0.5
dmp = 4.0/nx

x = np.linspace(-0.5 * lx,0.5*lx,nx)
y = np.linspace(-0.5 * ly,0.5*ly,ny)
x,y = np.meshgrid(x,y,indexing='ij')
dx = lx/(nx-1)
dy = ly/(ny-1)
dt = cfl *min(dx,dy)/np.sqrt((k+4.0*G/3)/rho)

p0 =1.0
p = p0*np.exp(-x*x-y*y)
tauxx = np.zeros((nx,ny))
tauyy = np.zeros((nx,ny))
tauxy = np.zeros((nx-1,ny -1))
vx= np.zeros((nx+1,ny))
vy= np.zeros((nx,ny + 1))


fig, graph = plt.subplots(2, 2)
gr = []
gr.append(make_plot(fig, graph[0, 0], x, y, p, 'p'))
gr.append(make_plot(fig, graph[0, 1], x, y, tauxx, 'tau_xx'))
gr.append(make_plot(fig, graph[1, 0], x, y, tauyy, 'tau_yy'))
gr.append(make_plot(fig, graph[1, 1], x[:-1, :-1], y[:-1, :-1], tauxy, 'tau_xy'))


# ACTION LOOP
def ActionLoop(i):
  
  for j in range(5):
    div_v = np.diff(vx,1,0)/dx +np.diff(vy,1,1)/dy
    global p,tauxx,tauyy,tauxy
    p = p - div_v *k *dt
    tauxx = tauxx + (np.diff(vx,1,0)/dx - div_v/3)*2*G*dt
    tauyy = tauyy + (np.diff(vy,1,1)/dx - div_v/3)*2*G*dt
    tauxy = tauxy + (np.diff(vx[1:-1,:],1,1)/dy +np.diff(vy[:,1:-1],1,0)/dx)*G*dt
    dvxdt = (np.diff(-p[:,1:-1] + tauxx[:,1:-1],1,0)/dx +np.diff(tauxy,1,1)/dy)/rho
    vx[1:-1,1:-1]= vx[1:-1,1:-1]*(1-dmp) +dvxdt*dt
    dvydt = (np.diff(-p[1:-1,:] + tauyy[1:-1,:],1,1)/dy +np.diff(tauxy,1,0)/dx)/rho
    vy[1:-1,1:-1]= vy[1:-1,1:-1]*(1-dmp) +dvydt*dt
   
  fig.suptitle(str((i+1)*5))
  gr[0].set_array(p)
  gr[1].set_array(tauxx)
  gr[2].set_array(tauyy)
  gr[3].set_array(tauxy)
  gr[0].set_clim([p.min(), p.max()])
  gr[1].set_clim([tauxx.min(), tauxx.max()])
  gr[2].set_clim([tauyy.min(), tauyy.max()])
  gr[3].set_clim([tauxy.min(), tauxy.max()])
  return gr
    
anim = ani.FuncAnimation(fig=fig, func=ActionLoop, frames=int(nsteps/5),interval=5, blit=True)
anim.save("p_tau.gif", writer=ani.PillowWriter(fps=60))