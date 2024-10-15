# Моделирование двойного математического маятника методом Эйлера и Рунге-Кутты #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import get_cmap

# PHYSICS #

L1 = 1
L2 = 1
M1 = 1
M2 = 1

G = 9.8


def Hamiltonian(s):
    q1,q2,p1,p2 = s

    f = L1*L2*(M1+M2*np.sin(q1-q2)**2)

    return (M2*L2**2*p1**2 + (M1+M2)*L1**2*p2**2 - 2*M2*L1*L2*p1*p2*np.cos(q1-q2) )/(2*M2*L1*L2*f) - \
       (M1+M2)*G*L1*np.cos(q1) - M2*G*L2*np.cos(q2)

# Решаем уравнение s' = D(s) #

def D(t, s):
    q1,q2,p1,p2 = s

    f = L1*L2*(M1+M2*np.sin(q1-q2)**2)

    h1 = p1*p2*np.sin(q1-q2)/f
    h2 = (M2*L2**2*p1**2 + (M1+M2)*L1**2*p2**2 - 2*M2*L1*L2*p1*p2*np.cos(q1-q2))/(2*f**2)

    return np.array((
        (L2*p1 - L1*np.cos(q1-q2)*p2 )/(L1*f),

        (-M2*L2*np.cos(q1-q2)*p1 + (M1+M2)*L1*p2)/(M2*L2*f),

        -(M1+M2)*G*L1*np.sin(q1) - h1 + h2*np.sin(2*(q1-q2)),

        -M2*G*L2*np.sin(q2) + h1 - h2*np.sin(2*(q1-q2))
    ))

PendelumsCount = 100

S_0 = np.zeros((4,PendelumsCount))

Q1_0 = np.linspace(3,3,PendelumsCount)
Q2_0 = np.linspace(1.5,1.50001,PendelumsCount)

S_0[0] = Q1_0
S_0[1] = Q2_0

# NUMERIC #

FPS = 20

DeltaT = 0.05

T_min, T_max = 0, 100

StepsNumber = int( (T_max-T_min) * FPS )

TimeAxes = np.linspace(T_min, T_max, StepsNumber+1)

StateHistory = np.zeros((4, PendelumsCount, StepsNumber+1))

import time

t = time.time()

## Prosessing #

# for i in range(PendelumsCount):
#     StateHistory[:,i] = solve_ivp(fun=D, t_span=(T_min, T_max), y0=S_0[:,i], method='RK45', t_eval=TimeAxes, first_step=DeltaT, max_step=DeltaT).y

StateHistory[:,:,0] = S_0

def RK4_step(s):
    d1 = DeltaT*D(0,s)
    d2 = DeltaT*D(0,s+d1/2)
    d3 = DeltaT*D(0,s+d2/2)
    d4 = DeltaT*D(0,s+d3)

    return s + ( d1 + 2*d2 + 2*d3 + d4 )/6

for i in range(StepsNumber):
    StateHistory[:,:,i+1] = RK4_step(StateHistory[:,:,i])

print(time.time()-t)

# Enegry = Hamiltonian(StateHistory)

# plt.plot(TimeAxes, Enegry)
# plt.show()

# plt.plot(TimeAxes, StateHistory[0])
# plt.plot(TimeAxes, StateHistory[1])
# plt.plot(TimeAxes, PEnegry)
# plt.plot(TimeAxes, LEnergy)

# plt.plot(TimeAxes, StateHistory[:,3])

# plt.show()



def show_pendulum_move(StateHistory):

    Q1History = StateHistory[0]
    Q2History = StateHistory[1]

    Fig, Ax = plt.subplots()

    CMap = get_cmap('copper')
    Colors = [CMap(j / (PendelumsCount + 1)) for j in range(PendelumsCount)]  

    Lines = [Ax.plot([], [], lw=1,marker=',', color=Colors[j])[0] for j in range(PendelumsCount)]


    # PendeliumPlot = Ax.plot(Pendelium[1], Pendelium[0],  marker='o')[0]

    Ax.set_xlim(-2.2, 2.2)
    Ax.set_ylim(-2.2, 2.2)
    Ax.grid(color = 'black', alpha = 0.25)
    Ax.set(xlabel='X', ylabel='Y', )
    Ax.legend()


    def loop_animation(i):
        for j in range(PendelumsCount):
            Lines[j].set_data([
                0, L1*np.sin(Q1History[j,i]), L1*np.sin(Q1History[j,i]) + L2*np.sin(Q2History[j,i])
            ], [
                0, -L1*np.cos(Q1History[j,i]), -L1*np.cos(Q1History[j,i]) - L2*np.cos(Q2History[j,i])
            ])

        return Lines

    ani = animation.FuncAnimation(
        fig=Fig, 
        func=loop_animation, 
        frames=StepsNumber, 
        interval=1000/FPS
    )
    # ani.save("double_pendulum2.gif", writer='pillow')
    plt.show()

show_pendulum_move(StateHistory)