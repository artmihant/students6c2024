import numpy as np
import matplotlib.pyplot as plt

U_0 = 0
V_0 = 1
DeltaT = 0.1
T_min, T_max = 0, 10
StepsNumber = int((T_max - T_min) / DeltaT)
    
n = 10
omega_0 = 1.0 
omega_1 = 2.0
Gamma = 0

omegas = np.linspace(omega_0, omega_1, n)
lengths = [9.81 / omega**2 for omega in omegas]  

def D(s, Gamma, Omega):
    u = s[0]
    v = s[1]
    return np.array((v, -2 * Gamma * v - Omega**2 * np.sin(u)))

def euler_step_rk4(s, Gamma, Omega):
    d1 = DeltaT*D(s, Gamma, Omega)
    d2 = DeltaT*D(s+d1/2, Gamma, Omega)
    d3 = DeltaT*D(s+d2/2, Gamma, Omega)
    d4 = DeltaT*D(s+d3, Gamma, Omega) 
    return s + d1/6+d2/3+d3/3+d4/6
    
def simulate_trajectory(Gamma,Omega):
    STrajectory = np.zeros((StepsNumber + 1, 2))
    STrajectory[0][0] = U_0
    STrajectory[0][1] = V_0
    for i in range(1, StepsNumber + 1):
        STrajectory[i] = euler_step_rk4(STrajectory[i - 1], Gamma, Omega)
    return STrajectory

def show_n(UTrajectories, lengths):
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    lines = []

    max_length = max(lengths)
    plt.xlim(-max_length - 0.2, max_length + 0.2)
    plt.ylim(-max_length - 0.2, max_length + 0.2)

    ax.set(xlabel='X', ylabel='Y')

    for i in range(len(UTrajectories)):
        x_coords = -lengths[i] * np.cos(UTrajectories[i])
        y_coords = lengths[i] * np.sin(UTrajectories[i])
        line, = ax.plot(y_coords, x_coords, marker='o', label=f'Маятник {i + 1}')
        lines.append((line, (x_coords, y_coords)))

    ax.legend()

    def loop_animation(i):
        for line, (x_coords, y_coords) in lines:
            line.set_data(y_coords[i:i+1], x_coords[i:i+1])
        return [line for line, _ in lines]


    ani = animation.FuncAnimation(
        fig=fig,
        func=loop_animation,
        frames=StepsNumber,
        interval=40,
        repeat=True,
        repeat_delay=0
    )
    #ani.save(filename="pendulum_animation.gif", writer='pillow')
    plt.show()


trajectories = []
for omega in omegas:
    STrajectory = simulate_trajectory(Gamma,omega)
    trajectories.append(STrajectory[:, 0]) 

show_n(trajectories, lengths)
