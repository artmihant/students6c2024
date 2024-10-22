import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm




M1 = 1
M2 = 1
G = 9.8

n = 100
L1_min = 0.15
L1_max = 1

m = 100
L2_min = 0.15
L2_max = 1

L1_values = np.linspace(L1_min, L1_max, n)
L2_values = np.linspace(L2_min, L2_max, m)

PendelumsCount = n * m
S_0 = np.zeros((6, PendelumsCount))

q1_0 = np.pi / 4
q2_0 = np.pi / 4
p1_0 = 1
p2_0 = 1

S_0[0] = q1_0
S_0[1] = q2_0
S_0[2] = p1_0
S_0[3] = p2_0
S_0[4] = np.repeat(L1_values, m)  
S_0[5] = np.tile(L2_values, n)    

def Lagrangian(s):
    q1, q2, p1, p2, L1, L2 = s
    U = G * ((M1 + M2) * L1 * np.cos(q1) + M2 * L2 * np.cos(q2))
    return Hamiltonian(s) - 2 * U

def Hamiltonian(s):
    q1, q2, p1, p2, L1, L2 = s
    f = L1 * L2 * (M1 + M2 * np.sin(q1 - q2)**2)
    return (M2 * L2**2 * p1**2 + (M1 + M2) * L1**2 * p2**2 - 2 * M2 * L1 * L2 * p1 * p2 * np.cos(q1 - q2)) / (2 * M2 * L1 * L2 * f) - \
           (M1 + M2) * G * L1 * np.cos(q1) - M2 * G * L2 * np.cos(q2)

def D(t, s):
    q1, q2, p1, p2, L1, L2 = s
    f = L1 * L2 * (M1 + M2 * np.sin(q1 - q2)**2)
    h1 = p1 * p2 * np.sin(q1 - q2) / f
    h2 = (M2 * L2**2 * p1**2 + (M1 + M2) * L1**2 * p2**2 - 2 * M2 * L1 * L2 * p1 * p2 * np.cos(q1 - q2)) / (2 * f**2)
    return np.array((
        (L2 * p1 - L1 * np.cos(q1 - q2) * p2) / (L1 * f),
        (-M2 * L2 * np.cos(q1 - q2) * p1 + (M1 + M2) * L1 * p2) / (M2 * L2 * f),
        -(M1 + M2) * G * L1 * np.sin(q1) - h1 + h2 * np.sin(2 * (q1 - q2)),
        -M2 * G * L2 * np.sin(q2) + h1 - h2 * np.sin(2 * (q1 - q2)),
        np.zeros(PendelumsCount),  
        np.zeros(PendelumsCount)   
    ))

FPS = 20
DeltaT = 0.05
T_min, T_max = 0, 5
StepsNumber = int((T_max - T_min) * FPS)
TimeAxes = np.linspace(T_min, T_max, StepsNumber + 1)

StateHistory = np.zeros((6, PendelumsCount, StepsNumber + 1))

StateHistory[:, :, 0] = S_0

def RK4_step(s):
    d1 = DeltaT * D(0, s)
    d2 = DeltaT * D(0, s + d1 / 2)
    d3 = DeltaT * D(0, s + d2 / 2)
    d4 = DeltaT * D(0, s + d3)
    return s + (d1 + 2 * d2 + 2 * d3 + d4) / 6

for i in range(StepsNumber):
    StateHistory[:, :, i + 1] = RK4_step(StateHistory[:, :, i])

def show_pendulum_move(StateHistory):
    L = np.zeros((n, m, StepsNumber + 1))
    for i in range(StepsNumber + 1):
        L[:, :, i] = Lagrangian(StateHistory[:, :, i]).reshape((n, m))

    Fig, Ax = plt.subplots()
    CMap = plt.get_cmap('copper')
    L1_grid, L2_grid = np.meshgrid(L1_values, L2_values)

    graph = Ax.pcolormesh(L1_grid, L2_grid, L[:, :, 0], cmap=CMap)
    Ax.set_title('Лагранжиан')
    Ax.set_xlabel('Длина L1')
    Ax.set_ylabel('Длина L2')
    Ax.set_xlim(L1_min, L1_max)  
    Ax.set_ylim(L2_min, L2_max)  
    
    def loop_animation(i):
        graph.set_array(L[:, :, i])
        return graph,

    ani = animation.FuncAnimation(
        fig=Fig,
        func=loop_animation,
        frames=StepsNumber,
        interval=1000 / FPS
    )
    ani.save(filename="pendulum_animation.gif", writer='pillow')
    plt.show()

show_pendulum_move(StateHistory)
