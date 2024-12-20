from matplotlib import pyplot as plt
import numpy as np

EI = 0.05
L_max = 1
q = 1

def Q(L):
    return (L_max - L)*q

Momentum = q*L_max**2/2 

def D(s):
    t, Phi, dPhi = s
    return np.array([1, dPhi, - Q(t)*np.cos(Phi)/EI])


N = 100
dL = L_max/N
eps = 1/(N**4) 

def RK4_step(s):
    d1 = dL*D(s)
    d2 = dL*D(s+d1/2)
    d3 = dL*D(s+d2/2)
    d4 = dL*D(s+d3)
    return s + ( d1 + 2*d2 + 2*d3 + d4 )/6

def T(M):
    state = np.array([0, 0, M])
    for _ in range(N):
        state = RK4_step(state)
    return state[2]
    
    
# 1) Метод стрельбы с использованием метода секущих - то что было на паре


def shooting():
    M_A, M_B = Momentum/EI, Momentum/(2*EI)
    T_A, T_B = T(M_A), T(M_B)

    while True:
        if abs(T_A) < eps:
            return M_A
        if abs(T_B) < eps:
            return M_B

        if abs(T_A - T_B) < eps:
            raise ValueError("Неудачный выбор начальных точек! Метод не сошелся.")

        M_C = (T_A*M_B - T_B*M_A)/(T_A - T_B)
        T_C = T(M_C)
        if T_A*T_B > 0:
            if abs(M_C-M_A) > abs(M_C-M_B):
                M_A, T_A = M_C, T_C
            else:
                M_B, T_B = M_C, T_C
        else:
            if T_A*T_C > 0:
                M_A, T_A = M_C, T_C
            else:
                M_B, T_B = M_C, T_C


# 2) Метод Ньютона с использованием центральной разности для T'(M)


def newton():
    M_curr = Momentum/EI/2 		# начальное приближение
    delta = 1e-6  				# dM
    max_iter=100
    for iteration in range(max_iter):
        T_curr = T(M_curr)
        if abs(T_curr) < eps:
            print("Метод Ньютона сошелся за", iteration, "итераций")
            return M_curr

        T_derivative = (T(M_curr + delta) - T(M_curr - delta)) / (2*delta)

        if abs(T_derivative) < 1e-10:
            raise ValueError(" метод разошелся ")

        M_next = M_curr - T_curr/T_derivative
        M_curr = M_next

    raise ValueError("Метод Ньютона не сошелся за", max_iter, "итераций")


# Сравнение методов

M_shooting = shooting()
M_newton = newton()

print("Найденное M методом секущих:", M_shooting)
print("Найденное M методом Ньютона:", M_newton)

def solve_trajectory(M):
    StateTrajectory = np.zeros((N+1, 3))
    StateTrajectory[0] = np.array([0, 0, M])
    for i in range(N):
        StateTrajectory[i+1] = RK4_step(StateTrajectory[i])
    return StateTrajectory

Trajectory_shooting = solve_trajectory(M_shooting)
Trajectory_newton = solve_trajectory(M_newton)

PhiTrajectory_shooting = Trajectory_shooting[:,1]
PhiTrajectory_newton = Trajectory_newton[:,1]

XTrajectory_shooting = np.cumsum(L_max*np.cos(PhiTrajectory_shooting)*dL) 
YTrajectory_shooting = np.cumsum(-L_max*np.sin(PhiTrajectory_shooting)*dL)

XTrajectory_newton = np.cumsum(L_max*np.cos(PhiTrajectory_newton)*dL) 
YTrajectory_newton = np.cumsum(-L_max*np.sin(PhiTrajectory_newton)*dL)

fig, ax  = plt.subplots()
margin = 0.01
ax.set_xlim(left=-margin, right=L_max/2+margin)
ax.set_ylim(top=margin, bottom=-L_max-margin)

ax.plot(XTrajectory_shooting, YTrajectory_shooting, label="Метод секущих")
ax.plot(XTrajectory_newton, YTrajectory_newton, label="Метод Ньютона", linestyle='--')

ax.legend()
plt.show()
