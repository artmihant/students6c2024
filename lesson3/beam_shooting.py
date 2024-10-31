from matplotlib import pyplot as plt
import numpy as np

"""
    Моделируется статическая конечная деформация закрепленной консольной балки 
    Подобная деформация подчиняется закону E*I*Phi'' = - Q(l)*cos(Phi) где 
    Phi(l) - угол наклона касательной к средней линии балки,
    l - параметр длины вдоль центральной линии от точки закрепления до текущей
    x(l), y(l) - горизонтальная и вертикальная координата компонент балки
    x'(l) = cos(Phi), y'(l) = sin(Phi), продольным растяжением звеньев пренебрегаем
    Е - модуль Юнга материала балки, I - статический момент сечения
    Q(l) = функция веса куска балки от точки t до её свободного конца
    q - распределенная сила (тяжести), l_max - длина балки
    если q не зависит от t, Q(t) = (L-l)*q
    сила тяжести считается направленной вдоль y в положительном направлении
    q,E,I вообще говоря могут зависеть от t; в нашем примере они константны
    Отметим, что момент балки в точке t равен E*I*Phi'(t) 
"""

## Задаем константы и параметры задачи ##

EI = 0.05
L_max = 1
q = 1

def Q(L):
    """ мы можем управлять здесь распределением нагрузки """
    # return q*L_max   # например сосредоточить всё на свободный конец
    return (L_max-L)*q # или распределить её равномерно

Momentum = q*L_max**2/2 # момент выпрямленной горизонтально балки при таком Q

def D(s):
    t, Phi, dPhi = s
    return np.array([1, dPhi, - Q(t)*np.cos(Phi)/EI])

"""
    Наша задача имеет граничные условия: Phi(0) = 0; Phi'(L_max) = 0: это краевая задача.
    Метод стрельбы применим для решения одномерных краевых задач
    Идея метода - подобрать M = Phi'(0) - недостающее до задачи Коши условие в точке 0 так, 
    что бы невязка T(M) была примерно равна 0 
    В качестве невязки используем второе граничное условие Phi'(L_max)
"""


## Задаем численные параметры метода и определим функцию невязки T(M) ##

N = 100

dL = L_max/N

eps = 1/(N**4) # глобальная точность, обеспечивающаясяя RK4

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

""" 
    Будем пытаться попасть в цель | подобрать M методом секущих с выбором точке
    Для метода нам требуются две точки A:(M_A, T_A) и B:(M_B,T,B) на плоскости [M,T(M)]
    На каждой итерации вычисляем третью точку С: (M_С, T_С) 
    посредством проведения прямой АВ и взятием M_C как пересечения этой прямой с абциссой,
    и вычислением T_C = T(M_C)
    Затем заменяем одну из точек А, B на С, так что бы (по возможности) 
    новые T_A, T_B имели разный знак, а если не получится, 
    пусть M_A и M_B будут ближе друг-к-другу

    Для первых двух приближений M_A, M_B заметим, что Momentum >= E*I*Phi'(0) 
    выразим M_A, M_B через Momentum и половину его.

"""

## Определим функцию стрельбы ##

def shooting():

    M_A, M_B = Momentum/EI, Momentum/(2*EI)
    T_A, T_B = T(M_A), T(M_B)

    print(M_A, T_A)
    print(M_B, T_B)

    while True:
        if abs(T_A) < eps:
            return M_A

        if abs(T_B) < eps:
            return M_B

        if abs(T_A - T_B) < eps:
            raise ValueError('Неудачный выбор начальных точек! Метод не сошелся.')

        M_C = (T_A*M_B - T_B*M_A)/(T_A-T_B)

        T_C = T(M_C)

        print(M_C, T_C)

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


M = shooting()

StateTrajectory = np.zeros((N+1, 3))

StateTrajectory[0] = np.array([0, 0, M])

for i in range(N):
    StateTrajectory[i+1] = RK4_step(StateTrajectory[i])

## Смоделируем финальную форму балки:

LTrajectory = StateTrajectory[:,0]
PhiTrajectory = StateTrajectory[:,1]

XTrajectory = np.cumsum(L_max*np.cos(PhiTrajectory)*dL) # частичные суммы
YTrajectory = np.cumsum(-L_max*np.sin(PhiTrajectory)*dL)

fig, ax  = plt.subplots()

margin = 0.01

ax.set_xlim(left=-L_max-margin, right=L_max+margin)
ax.set_ylim(top=L_max+margin, bottom=-L_max-margin)

ax.plot(XTrajectory, YTrajectory)
# ax.margins(10, 10)

plt.show()


""" 
    Мы установили довольно слабый параметр жесткости EI , отчего наша балка получилась очень сильно обвисшей
    Поигравшись с программой и ещё ослабив EI и/или увеличив момент балки, можно добиться того, 
    что бы алгоритм нашел другую усточивую форму и балка перехлестнулась через точку подвеса, 
    свесившись с другой стороны (попробуйте сделать это!)
"""