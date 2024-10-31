
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import get_cmap

# PHYSICS #

L1 = 1
L2 = 1

Q1 = np.pi/4.
Q2 = np.pi/4.

G = 9.8



def Lagrangian(s):
    M1,M2,q1,q2,p1,p2 = s

    U = G*((M1+M2)*L1*np.cos(q1) + M2*L2*np.cos(q2))

    return Hamiltonian(s) - 2*U


def Hamiltonian(s):
    M1,M2,q1,q2,p1,p2 = s

    f = L1*L2*(M1+M2*np.sin(q1-q2)**2)

    return (M2*L2**2*p1**2 + (M1+M2)*L1**2*p2**2 - 2*M2*L1*L2*p1*p2*np.cos(q1-q2) )/(2*M2*L1*L2*f) - \
       (M1+M2)*G*L1*np.cos(q1) - M2*G*L2*np.cos(q2)



def D(t, s):
    M1,M2,q1,q2,p1,p2 = s

    f = L1*L2*(M1+M2*np.sin(q1-q2)**2)

    h1 = p1*p2*np.sin(q1-q2)/f
    h2 = (M2*L2**2*p1**2 + (M1+M2)*L1**2*p2**2 - 2*M2*L1*L2*p1*p2*np.cos(q1-q2))/(2*f**2)

    return np.array((
        (M1-M1),
        (M2-M2),
        (L2*p1 - L1*np.cos(q1-q2)*p2 )/(L1*f),

        (-M2*L2*np.cos(q1-q2)*p1 + (M1+M2)*L1*p2)/(M2*L2*f),

        -(M1+M2)*G*L1*np.sin(q1) - h1 + h2*np.sin(2*(q1-q2)),

        -M2*G*L2*np.sin(q2) + h1 - h2*np.sin(2*(q1-q2))
    ))

PendelumsWidth = 80
PendelumsHeight = 60
PendelumsCount = PendelumsWidth*PendelumsHeight

S_0 = np.zeros((6, PendelumsCount))

M1_min, M1_max = 0.1, 1.0
M2_min, M2_max = 0.1, 1.0

M1_0 = (np.zeros(PendelumsWidth).reshape((-1,1)) + np.linspace(M1_min, M1_max,PendelumsHeight).reshape((1,-1))).reshape(-1)
M2_0 = (np.zeros(PendelumsHeight).reshape((1,-1)) + np.linspace(M2_min, M2_max,PendelumsWidth).reshape((-1,1))).reshape(-1)

Q1_0 = (np.zeros(PendelumsWidth).reshape((-1,1)) + Q1*np.ones(PendelumsHeight).reshape((1,-1))).reshape(-1)
Q2_0 = (np.zeros(PendelumsHeight).reshape((1,-1)) + Q2*np.ones(PendelumsWidth).reshape((-1,1))).reshape(-1)

S_0[0] = M1_0
S_0[1] = M2_0

S_0[2] = Q1_0
S_0[3] = Q2_0



# NUMERIC #

FPS = 20

DeltaT = 0.05

T_min, T_max = 0, 10

StepsNumber = int( (T_max-T_min) * FPS )

TimeAxes = np.linspace(T_min, T_max, StepsNumber+1)

StateHistory = np.zeros((6, PendelumsCount, StepsNumber+1))

import time

t = time.time()

## Prosessing #

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


def show_pendulum_move(StateHistory):

    L = np.transpose(Lagrangian(StateHistory).reshape((PendelumsWidth,PendelumsHeight, StepsNumber+1)), [1,0,2])

    Fig, Ax = plt.subplots()

    CMap = get_cmap('copper')

    Lines = [Ax.plot([], [], lw=1,marker=',', color='black')[0] for j in range(PendelumsCount)]

    graph = Ax.pcolormesh(L[:,:,0])

    Ax.legend()

    def loop_animation(i):

        graph.set_array(L[:,:,i])

        return graph

    ani = animation.FuncAnimation(
        fig=Fig, 
        func=loop_animation, 
        frames=StepsNumber, 
        interval=1000/FPS
    )
    ani.save(filename="multy_frac_double_pendulum.gif", writer="imagemagick")

show_pendulum_move(StateHistory)