
import numpy as np

# PHYSICS #
N = 10
Gamma = [0] * N
Omega = [2,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1]
L     = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.7,0.9,1])

def D(s,gamma,omega):
    u = s[0]
    v = s[1]
    return np.array(( v, -2*gamma*v-omega**2*np.sin(u)))

U_0 = [1] * N
V_0 = [0] * N

# NUMERIC #

DeltaT = 0.1

T_min, T_max = 0, 10

StepsNumber = int( (T_max-T_min) / DeltaT )

# 

STrajectory = np.zeros((N , 2))

for i in range(N):
    STrajectory[i][0] = U_0[i]
    STrajectory[i][1] = V_0[i]

#def explicit_euler_step(s):
#    d = DeltaT*D(s)
#    # higher energy
#    return s + d

#def notexplicit_euler_step(s):
#    d1 = DeltaT*D(s)
#    d2 = DeltaT*D(s+d1)
    #return s + d2 - lower energy
#    return s + (d1 + d2) / 2

#def notexplicit_euler_step_with_MAGIC(s):
#    d1 = DeltaT*D(s)
#    d2 = DeltaT*D(s+d1/2)
#    return s + d2

def notexplicit_euler_step_with_MAGIC4(s,gamma,omega):
    d1 = DeltaT*D(s,gamma,omega)
    d2 = DeltaT*D(s+d1/2,gamma,omega)
    d3 = DeltaT*D(s+d2/2,gamma,omega)
    d4 = DeltaT*D(s+d3,gamma,omega)
    return s + d1/6 + d2/3 + d3/3 + d4/6

## Euler ##
#for i in range(1, StepsNumber+1 ):
#    for j in range(N):
#        STrajectory[N*i+j] = notexplicit_euler_step_with_MAGIC4(STrajectory[N*(i-1)+j],Gamma[j],Omega[j])


#UTrajectory = STrajectory[:, 0]
#VTrajectory = STrajectory[:, 1]

#ETrajectory = VTrajectory**2 + 2*Omega**2*(1-np.cos(UTrajectory))

import matplotlib.pyplot as plt

#fig, ax = plt.subplots()

#TimeAxes = np.linspace(T_min,T_max,StepsNumber+1)

#ax.plot(TimeAxes, UTrajectory, label='U')
#ax.plot(TimeAxes, VTrajectory, label='V')
#ax.set(xlabel='T', ylabel='U,V', title='U(t) и V(t)')

#plt.show()

#plt.plot(VTrajectory, UTrajectory)
#plt.plot(TimeAxes, ETrajectory)
#plt.show()

def show_pendulum_move():
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    # PhaseLine = ax.plot(Trajectory[0], Trajectory[1], label=f'Фазовая траектория маятника')[0]
    LineTrajectory = [0] * N
    for j in range(N):
        point = (L[j] * np.sin(STrajectory[j,0]), -L[j] * np.cos(STrajectory[j,0]))
        LineTrajectory[j] = ax.plot([0, point[0]], [0, point[1]], markevery=(1,1), marker='o')[0]

    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)

    ax.set(xlabel='X', ylabel='Y', )
    #ax.legend()

    def loop_animation(i):
        """ Главный цикл вычисления/анимации """
        for j in range(N):
            LineTrajectory[j].set_data([0, L[j] * np.sin(STrajectory[j,0])], [0, -L[j] * np.cos(STrajectory[j,0])])

        for j in range(N):
            STrajectory[j] = notexplicit_euler_step_with_MAGIC4(STrajectory[j],Gamma[j],Omega[j])

        return (LineTrajectory)

    ani = matplotlib.animation.FuncAnimation(
        fig=fig, 
        func=loop_animation, 
        frames=StepsNumber, 
        interval=40,
        repeat=True,
        repeat_delay=0
    )
    ani.save(filename="ans.gif", writer="imagemagick")
    #plt.show()

show_pendulum_move()














