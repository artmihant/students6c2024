# Моделирование двойного математического маятника методом Эйлера и Рунге-Кутты #
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import get_cmap

# PHYSICS #

choice = ""

# Choose you destiny:
if len(sys.argv) == 1:
    print("Not to choose is a choice too, i will choose Q for you, no need to thank me")
    choice = "Q"
else:
    choice = sys.argv[1]

L1_min = L1_max = 1
L2_min = L2_max = 1
M1_min = M1_max = 1
M2_min = M2_max = 1
Q1_min = Q1_max = np.pi/4
Q2_min = Q2_max = np.pi/4
P1_min = P1_max = 1
P2_min = P2_max = 1

match choice.lower():
    case "l":
        L1_min = 0.15; L1_max = 1
        L2_min = 0.15; L2_max = 1
    case "m":
        M1_min = 0.15; M1_max = 1
        M2_min = 0.15; M2_max = 1
    case "q":
        Q1_min = 0.5; Q1_max = 2
        Q2_min = 0.5; Q2_max = 2
    case "p":
        P1_min = 0.05; P1_max = 1
        P2_min = 0.05; P2_max = 1

PendelumsWidth = 60
PendelumsHeight = 60
PendelumsCount = PendelumsWidth*PendelumsHeight

S_0 = np.zeros((8, PendelumsCount))

#Q1_0 = (np.zeros(PendelumsWidth).reshape((-1,1)) + np.linspace(Q1_min, Q1_max,PendelumsHeight).reshape((1,-1))).reshape(-1)
#Q2_0 = (np.zeros(PendelumsHeight).reshape((1,-1)) + np.linspace(Q2_min, Q2_max,PendelumsWidth).reshape((-1,1))).reshape(-1)
# above can be replaced with numpy.repeat and numpy.tile, both repeat arrays but differently
#   >>> a = np.array([1,2,3])
#   >>> np.repeat(a,3)
#   array([1, 1, 1, 2, 2, 2, 3, 3, 3])
#   >>> np.tile(a,3)
#   array([1, 2, 3, 1, 2, 3, 1, 2, 3])
Q1_0 = np.tile(np.linspace(Q1_min, Q1_max,PendelumsHeight),PendelumsWidth)
Q2_0 = np.repeat(np.linspace(Q2_min, Q2_max,PendelumsWidth),PendelumsHeight)

M1_0 = np.tile(np.linspace(M1_min, M1_max,PendelumsHeight),PendelumsWidth)
M2_0 = np.repeat(np.linspace(M2_min, M2_max,PendelumsWidth),PendelumsHeight)

L1_0 = np.tile(np.linspace(L1_min, L1_max,PendelumsHeight),PendelumsWidth)
L2_0 = np.repeat(np.linspace(L2_min, L2_max,PendelumsWidth),PendelumsHeight)

P1_0 = np.tile(np.linspace(P1_min, P1_max,PendelumsHeight),PendelumsWidth)
P2_0 = np.repeat(np.linspace(P2_min, P2_max,PendelumsWidth),PendelumsHeight)

S_0[0] = Q1_0
S_0[1] = Q2_0
S_0[2] = M1_0
S_0[3] = M2_0
S_0[4] = L1_0
S_0[5] = L2_0
S_0[6] = P1_0
S_0[7] = P2_0

G = 9.8

def Lagrangian(s):
    q1,q2,M1,M2,L1,L2,p1,p2 = s
    U = G*((M1+M2)*L1*np.cos(q1) + M2*L2*np.cos(q2))
    return Hamiltonian(s) - 2*U

def Hamiltonian(s):
    q1,q2,M1,M2,L1,L2,p1,p2 = s

    f = L1*L2*(M1+M2*np.sin(q1-q2)**2)

    return (M2*L2**2*p1**2 + (M1+M2)*L1**2*p2**2 - 2*M2*L1*L2*p1*p2*np.cos(q1-q2) )/(2*M2*L1*L2*f) - \
       (M1+M2)*G*L1*np.cos(q1) - M2*G*L2*np.cos(q2)

# Решаем уравнение s' = D(s) #

def D(t, s):
    q1,q2,M1,M2,L1,L2,p1,p2 = s

    f = L1*L2*(M1+M2*np.sin(q1-q2)**2)

    h1 = p1*p2*np.sin(q1-q2)/f
    h2 = (M2*L2**2*p1**2 + (M1+M2)*L1**2*p2**2 - 2*M2*L1*L2*p1*p2*np.cos(q1-q2))/(2*f**2)

    return np.array((
        (L2*p1 - L1*np.cos(q1-q2)*p2 )/(L1*f),

        (-M2*L2*np.cos(q1-q2)*p1 + (M1+M2)*L1*p2)/(M2*L2*f),
        
        np.array([0]*PendelumsCount), # we add all those back later
        np.array([0]*PendelumsCount),
        np.array([0]*PendelumsCount),
        np.array([0]*PendelumsCount),

        -(M1+M2)*G*L1*np.sin(q1) - h1 + h2*np.sin(2*(q1-q2)),

        -M2*G*L2*np.sin(q2) + h1 - h2*np.sin(2*(q1-q2))
    ))

# NUMERIC #

FPS = 20

DeltaT = 0.05

T_min, T_max = 0, 10

StepsNumber = int( (T_max-T_min) * FPS )

#TimeAxes = np.linspace(T_min, T_max, StepsNumber+1)

StateHistory = np.zeros((8, PendelumsCount, StepsNumber+1))

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

    L = np.transpose(Lagrangian(StateHistory).reshape((PendelumsWidth,PendelumsHeight, StepsNumber+1)), [1,0,2])

    Fig, Ax = plt.subplots()

    CMap = get_cmap('copper')

    #Lines = [Ax.plot([], [], lw=1,marker=',', color='black')[0] for j in range(PendelumsCount)]

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
    ani.save("multy2_double_pendulum_lagrange_l.mkv", writer='ffmpeg')
    #plt.show()

show_pendulum_move(StateHistory)