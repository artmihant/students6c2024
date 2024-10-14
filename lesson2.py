# Моделирование двойного математического маятника методом Эйлера и Рунге-Кутты #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PHYSICS #

L1 = 1
L2 = 1
M1 = 1
M2 = 1

G = 1

def H(s):
    q1 = s[:,0]
    q2 = s[:,1]
    p1 = s[:,2]
    p2 = s[:,3]

    f = L1*L2*(M1+M2*np.sin(q1-q2)**2)

    return (M2*L2**2*p1**2 + (M1+M2)*L1**2*p2**2 - 2*M2*L1*L2*p1*p2*np.cos(q1-q2) )/(2*M2*L1*L2*f) - \
       (M1+M2)*G*L1*np.cos(q1) - M2*G*L2*np.cos(q2)

# Решаем уравнение s' = D(s) #

def D(s):
    q1 = s[0]
    q2 = s[1]
    p1 = s[2]
    p2 = s[3]

    f = L1*L2*(M1+M2*np.sin(q1-q2)**2)

    h1 = p1*p2*np.sin(q1-q2)/f
    h2 = (M2*L2**2*p1**2 + (M1+M2)*L1**2*p2**2 - 2*M2*L1*L2*p1*p2*np.cos(q1-q2))/(2*f**2)

    return np.array((
        (L2*p1 - L1*np.cos(q1-q2)*p2 )/(L1*f),

        (-M2*L2*np.cos(q1-q2)*p1 + (M1+M2)*L1*p2)/(M2*L2*f),

        -(M1+M2)*G*L1*np.sin(q1) - h1 + h2*np.sin(2*(q1-q2)),

        -M2*G*L2*np.sin(q2) + h1 - h2*np.sin(2*(q1-q2))
    ))

Q1_0 = 0
Q2_0 = 1
P1_0 = 0
P2_0 = 0

S_0 = np.array((Q1_0, Q2_0, P1_0, P2_0), dtype=np.float32)

# NUMERIC #

DeltaT = 0.1

T_min, T_max = 0, 100

StepsNumber = int( (T_max-T_min) / DeltaT )

# Make state history array

StateHistory = np.zeros((StepsNumber+1 , 4), dtype=np.float32)

StateHistory[0] = S_0

TimeAxes = np.linspace(T_min, T_max, StepsNumber+1)

# METHOD #

def RK4_step(s):
    #RK4
    d1 = DeltaT*D(s)
    d2 = DeltaT*D(s+d1/2)
    d3 = DeltaT*D(s+d2/2)
    d4 = DeltaT*D(s+d3)

    return s + ( d1 + 2*d2 + 2*d3 + d4 )/6


# Prosessing #

## Euler ##
for i in range(StepsNumber):
    StateHistory[i+1] = RK4_step(StateHistory[i])


Enegry = H(StateHistory)

plt.plot(TimeAxes, Enegry)

# plt.plot(TimeAxes, UTrajectory)
# plt.plot(TimeAxes, VTrajectory)

# plt.show()

def show_pendulum_move(StateHistory):

    Q1History = StateHistory[:,0]
    Q2History = StateHistory[:,1]

    Pendelium = np.zeros((2,3))

    Fig, Ax = plt.subplots()

    PendeliumPlot = Ax.plot(Pendelium[1], Pendelium[0],  marker='o')[0]

    Ax.set_xlim(-2.5, 2.5)
    Ax.set_ylim(-2.5, 2.5)

    Ax.set(xlabel='X', ylabel='Y', )
    Ax.legend()

    def loop_animation(i):
        """ Главный цикл вычисления/анимации """

        Pendelium[0][1] = -L1*np.cos(Q1History[i])
        Pendelium[1][1] = L1*np.sin(Q1History[i])
        Pendelium[0][2] = -L1*np.cos(Q1History[i]) - L2*np.cos(Q2History[i])    
        Pendelium[1][2] = L1*np.sin(Q1History[i]) + L2*np.sin(Q2History[i])  

        PendeliumPlot.set_data(Pendelium[1], Pendelium[0])

        return (PendeliumPlot)

    ani = animation.FuncAnimation(
        fig=Fig, 
        func=loop_animation, 
        frames=StepsNumber, 
        interval=DeltaT*100
    )
    plt.show()

show_pendulum_move(StateHistory)