"""
Сравнение численного решения уравнения теплопроводности с аналитическим
на грфике справа выводится изменение L2 номры разности решений со временем
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.animation import PillowWriter
lx = 10.0
lam = 1.0
rhocp = 1.0
T0 = 1.0

nx = 100
nt = 400
cfl = 1.0

x = np.linspace(0, lx, nx)
dx = lx / (nx - 1)
dt = dx**2 * rhocp / lam * 0.5 * cfl

t = 0.0
T = T0 * np.sin(np.pi * x / lx)
qx = np.zeros(nx - 1)

def heat_eq_solution(x, t):
    #удовлетворяет граничным условиям T(x=0)=T(x=lx)=0 - значение в этих точках в нач момент
	#удовлетворяет уравнени. T_t = lam/rhocp T_xx
    return np.sin(np.pi * x / lx) * np.exp(-(lam / rhocp) * (np.pi / lx)**2 * t)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

gr_num, = ax1.plot(x, T, label="Num. Temperature")
gr_an, = ax1.plot(x, heat_eq_solution(x, t), label="Analit. Temperature", linestyle='--')
ax1.set_xlabel('x')
ax1.set_ylabel('Temperature')
ax1.set_title('Temperature Distribution')
ax1.legend()

times = []
l2_values = []
gr_l2, = ax2.plot([], [], label="L2 Norm")
ax2.set_xlabel('time')
ax2.set_ylabel('L2 norm')
ax2.set_title('L2 norm vs Time')
ax2.legend()
ax2.set_xlim(0, nt*dt)
ax2.set_ylim(0, 0.0001)  

def action_loop(frame):
    global T, t
    t += dt
    qx[:] = -lam * np.diff(T) / dx
    dTdt = -np.diff(qx) / dx
    T[1:-1] += dTdt * dt / rhocp
    T_an = heat_eq_solution(x, t)
    L2_norm = np.sqrt(np.sum((T - T_an)**2) * dx)

    gr_num.set_ydata(T)
    gr_an.set_ydata(T_an)

    times.append(t)
    l2_values.append(L2_norm)
    gr_l2.set_xdata(times)
    gr_l2.set_ydata(l2_values)

    ax2.relim()
    ax2.autoscale_view()
    fig.suptitle(f"Time step: {frame+1}, t={t}, L2 norm: {L2_norm}")
    
    return gr_num, gr_an, gr_l2

animation = ani.FuncAnimation(fig, action_loop, frames=nt, interval=1, repeat=True)
animation.save("HEQ1D_compare_with_analyth.gif", writer=PillowWriter(fps=10))
plt.tight_layout()
plt.show()

