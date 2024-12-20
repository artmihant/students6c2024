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



## Задаем численные параметры метода  ##

N = 200
max_iter = 60

l = np.linspace(0, L_max, N)
h = L_max / (N - 1)
eps = 1/(N**4) # глобальная точность

PhiTrajectory = np.zeros(N)
 
#########

def R_and_J(Phi):   
    R = np.zeros(N) #Невязка
    J = np.zeros((N, N))#матрица Якоби
    for i in range(0, N):
        # Закрепленный конец
        if i == 0:
            R[i] = Phi[0]
            
            J[i, i] = 1
        # Свободный конец
        elif i == N-1:
            R[i] = (Phi[i] - Phi[i - 1]) / h
            
            J[i, i-1] = -1 / h
            J[i, i] = 1 / h
        # промежуточные точки
        else:
            R[i] = (Phi[i-1] - 2 * Phi[i] + Phi[i+1]) / h**2 + (Q(l[i]) / EI) * np.cos(PhiTrajectory[i])
            
            J[i, i - 1] = 1 / h**2
            J[i, i]     = -2 / h**2 - (Q(l[i]) / EI) * np.sin(PhiTrajectory[i])
            J[i, i + 1] = 1 / h**2
    return R, J
# Цикл
for iteration in range(max_iter):
    R, J = R_and_J(PhiTrajectory)
    
    R_norm = np.linalg.norm(R, np.inf)
    print(f"На шаге {iteration} норма невязки = {R_norm}")
    if R_norm < eps:
        break
    # Решаем систему линейных уравнений
    delta = np.linalg.solve(J, R)
    # Обновляем значения
    PhiTrajectory[:] -= delta    
else:
    print("Метод Ньютона не сошелся за заданное число итераций")


## Смоделируем финальную форму балки:


XTrajectory = np.cumsum(L_max*np.cos(PhiTrajectory)*h) # частичные суммы
YTrajectory = np.cumsum(-L_max*np.sin(PhiTrajectory)*h)

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