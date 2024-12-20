import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import get_cmap

G = 9.81

PendulumNumber = 5

T_max   = 20

DeltaT      = 0.05
StepsNumber = int(T_max / DeltaT)

PendulumLengths =np.arange(3,4.5,0.3)
StateHistory = np.zeros((StepsNumber+1, 2, PendulumNumber)) 
StateHistory[0, 0] = np.pi / 3  # Начальный угол отклонения


def D(s):
    u = s[0]
    v = s[1]
    return np.array(( v, -(G / PendulumLengths) * np.sin(u)))


def RK4_step(s):
    d1 = DeltaT*D(s)
    d2 = DeltaT*D(s+d1/2)
    d3 = DeltaT*D(s+d2/2)
    d4 = DeltaT*D(s+d3)
    return s + ( d1 + 2*d2 + 2*d3 + d4 )/6

for i in range(StepsNumber):
    StateHistory[i+1] = RK4_step(StateHistory[i])

# Animation block

AnglesHistory = StateHistory[:,0]

Fig, Ax = plt.subplots()
Ax.set_xlim(-5, 6)
Ax.set_ylim(-5, 0.5)
Ax.grid(color = 'black', alpha = 0.25)

# Цвета маятников
CMap = get_cmap('Spectral')
Colors = [CMap(i / (PendulumNumber - 1)) for i in range(PendulumNumber)]  

Lines = [Ax.plot([], [], lw=1, color=Colors[i])[0] for i in range(PendulumNumber)]
Bobs = [Ax.plot([], [], 'o', color=Colors[i], markersize=12)[0] for i in range(PendulumNumber)]

def init():
    for line, bob in zip(Lines, Bobs):
        line.set_data([], [])
        bob.set_data([], [])
    return Lines + Bobs

def update(frame):
    for j in range(PendulumNumber):
        x = PendulumLengths[j] * np.sin(AnglesHistory[frame, j])
        y = -PendulumLengths[j] * np.cos(AnglesHistory[frame, j])        
        Lines[j].set_data([0, x], [0, y])
        Bobs[j].set_data(x, y)
    return Lines + Bobs

Animation = FuncAnimation(Fig, update, frames=StepsNumber, init_func=init, blit=True, interval=DeltaT*1000)

print('Save GIF')
Animation.save(filename="5pendulum.gif", writer="pillow")
