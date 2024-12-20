import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

X_min, X_max = -2, 1
Y_min, Y_max = -1.5, 1.5

X_step = 1000
Y_step = 1000

IterMax = 100

XSpace = np.linspace(X_min, X_max, X_step).reshape((1,-1))
YSpace = np.linspace(Y_min, Y_max, Y_step).reshape((-1,1))
C = XSpace + 1j*YSpace  

IterCount = np.zeros(C.shape, dtype=int) #
Z = np.zeros(C.shape, dtype=complex)
IterHistory = []

for i in range(IterMax):
    Z = Z*Z + C
    escaped = np.abs(Z) > 4								#смотрим какие вышли за круг радиуса 4
    IterCount[escaped & (IterCount == 0)] = i			#для тех кто вышли только на этом шаге запоминаем номер шага
    IterHistory.append(IterCount.copy())				

fig, ax = plt.subplots()
im = ax.imshow(IterHistory[0],cmap='hot',extent=(X_min, X_max, Y_min, Y_max),vmin=0,vmax=100)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_title('Множество Мандельброта')
cb = plt.colorbar(im, ax=ax, label='итераций до abs(z)>4')

def update_frame(frame):
    im.set_data(IterHistory[frame])
    ax.set_title(f'Множество Мандельброта (итерация {frame+1})')
    return [im]
ani = animation.FuncAnimation(fig, update_frame, frames=IterMax, interval=10, repeat=True)
ani.save('mandelbrot.gif', writer='imagemagick', fps=10)
plt.show()
