import numpy as np
import matplotlib.pyplot as plt

# PHYSICS #
N = 7
Gamma = [0] * N
Omega = np.random.uniform(low=1.0, high = 2.0, size= N)
L = np.random.uniform(low=0.1, high = 1.0, size= N)

def D(s,gamma,omega):
    u = s[0]
    v = s[1]
    return np.array(( v, -2*gamma*v-omega**2*np.sin(u)))

U_0 = [1] * N
V_0 = [0] * N

# NUMERIC #

DeltaT = 0.1

T_min, T_max = 0, 20

StepsNumber = int( (T_max-T_min) / DeltaT )

# 

STrajectory = np.zeros((N , 2))

for i in range(N):
    STrajectory[i][0] = U_0[i]
    STrajectory[i][1] = V_0[i]


def notexplicit_euler_step_with_MAGIC4(s,gamma,omega):
    d1 = DeltaT*D(s,gamma,omega)
    d2 = DeltaT*D(s+d1/2,gamma,omega)
    d3 = DeltaT*D(s+d2/2,gamma,omega)
    d4 = DeltaT*D(s+d3,gamma,omega)
    return s + d1/6 + d2/3 + d3/3 + d4/6


def show_pendulum_move():
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    LineTrajectory = [0] * N
    for j in range(N):
        point = (L[j] * np.sin(STrajectory[j,0]), -L[j] * np.cos(STrajectory[j,0]))
        LineTrajectory[j] = ax.plot([0, point[0]], [0, point[1]], markevery=(1,1), marker='o')[0]
    
    ax.legend(['L = %.1f, w =%.1f' %(L[j], Omega[j]) for j in range(N)])

    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)

    ax.set(xlabel='X', ylabel='Y', )

    def loop_animation(i):
        for j in range(N):
            LineTrajectory[j].set_data([0, L[j] * np.sin(STrajectory[j,0])], [0, -L[j] * np.cos(STrajectory[j,0])])

        for j in range(N):
            STrajectory[j] = notexplicit_euler_step_with_MAGIC4(STrajectory[j],Gamma[j],Omega[j])

        return (LineTrajectory)

    ani = animation.FuncAnimation(
        fig=fig, 
        func=loop_animation, 
        frames=StepsNumber, 
        interval=40,
        repeat=True,
        repeat_delay=0
    )
    ani.save(filename="pendulum.gif", writer="imagemagick")

show_pendulum_move()