# Моделирование математического маятника методом Эйлера и Рунге-Кутты #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PHYSICS #

Gamma = 0
Omega = 1

def D(s):
    u = s[0]
    v = s[1]
    return np.array(( v, -2*Gamma*v-Omega**2*np.sin(u)))


U_0 = 0
V_0 = 2

S_0 = np.array((U_0, V_0), dtype=np.float32)

# NUMERIC #

DeltaT = 0.1

T_min, T_max = 0, 100

StepsNumber = int( (T_max-T_min) / DeltaT )

# Make state history array

STrajectory = np.zeros((StepsNumber+1 , 2), dtype=np.float32)

STrajectory[0] = S_0

TimeAxes = np.linspace(T_min, T_max, StepsNumber+1)

# METHOD #

def explicit_euler_step(s):
    d1 = DeltaT*D(s)
    return s + d1

def implicit_euler_step(s):
    d1 = DeltaT*D(s)
    d2 = DeltaT*D(s+d1)
    return s + d2

def average_exp_imp_step(s):
    #RK2
    d1 = DeltaT*D(s)
    d2 = DeltaT*D(s+d1)
    return s + (d1+d2)/2

def half_point_step(s):
    #RK2
    d1 = DeltaT*D(s)
    d2 = DeltaT*D(s+d1/2)
    return s + d2

def RK4_step(s):
    #RK4
    d1 = DeltaT*D(s)
    d2 = DeltaT*D(s+d1/2)
    d3 = DeltaT*D(s+d2/2)
    d4 = DeltaT*D(s+d3)

    return s + ( d1 + 2*d2 + 2*d3 + d4 )/6


# Prosessing #

## Euler ##
for i in range(1, StepsNumber+1 ):
    STrajectory[i] = RK4_step(STrajectory[i-1])


UTrajectory = STrajectory[:, 0]
VTrajectory = STrajectory[:, 1]

Enegry = VTrajectory**2 + Omega**2 * 2*(1-np.cos(UTrajectory))



# plt.plot(TimeAxes, Enegry)

# plt.plot(TimeAxes, UTrajectory)
# plt.plot(TimeAxes, VTrajectory)


fig, axs  = plt.subplots(3, layout='constrained')

axs[0].plot(TimeAxes, UTrajectory, label='U')
axs[0].plot(TimeAxes, VTrajectory, label='V')
axs[0].set(xlabel='T', ylabel='U,V', title='U(t) и V(t)')

axs[1].plot(TimeAxes, Enegry)
axs[1].set(xlabel='T', ylabel='E', title='E(t)')

axs[2].plot(UTrajectory, VTrajectory)
axs[2].set(xlabel='U', ylabel='V', title='Фазовая траектория')


def show_pendulum_move(UTrajectory):

    L = 1

    Trajectory = (-L*np.cos(UTrajectory), L*np.sin(UTrajectory))

    fig, ax = plt.subplots()
    # PhaseLine = ax.plot(Trajectory[0], Trajectory[1], label=f'Фазовая траектория маятника')[0]

    LineTrajectory = ax.plot(Trajectory[1], Trajectory[0],  marker='o')[0]

    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)

    ax.set(xlabel='X', ylabel='Y', )
    ax.legend()

    def loop_animation(i):
        """ Главный цикл вычисления/анимации """

        LineTrajectory.set_data(Trajectory[1][i:i+5], Trajectory[0][i:i+5])

        return (LineTrajectory)

    ani = animation.FuncAnimation(
        fig=fig, 
        func=loop_animation, 
        frames=StepsNumber, 
        interval=0.02,
        repeat=True,
        repeat_delay=1000
    )
    plt.show()

show_pendulum_move(UTrajectory)