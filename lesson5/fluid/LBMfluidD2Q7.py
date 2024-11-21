import time
import matplotlib.pyplot as plt, matplotlib.animation
import numpy as np
import numba as nb

""" Зададим константы и свойства задачи """

Viscosity = 0.02                                # вязкость жидкости
Height, Width = 80, 200,                          # размеры решетки

U0 = np.array([0.1, 0])                        # начальная и внешняя скорость (в махах)

def BarrierShape():
    """ Инициализируем форму барьера. Необходимо установить True там, где барьер """
    barrier = np.zeros((Height, Width), bool)

    # круг
    for y in range(barrier.shape[0]):
        for x in range(barrier.shape[1]):
            if (y - Height//2)**2 + (x - Height//2)**2 < (Height//10)**2:
                barrier[y,x] = True
    # хвост
    # barrier[(Height//2), ((Height//2)):((Height//2)+4*(Height//10))] = True

    return barrier


""" Зададим свойства шаблона решетки D2Q9 """

D = 2 # Мерность модели
Q = 7 # число точек в шаблоне

s3d2 = 3**0.5/2

V = np.array([
    [-0, 0],
    [ 1, 0],[ 0.5, s3d2],[-0.5, s3d2],[-1, 0],[ -0.5, -s3d2],[ 0.5, -s3d2],
])

W = np.array([
    1/2, 
    1/12, 1/12, 1/12, 1/12, 1/12, 1/12
])

C = 1/2  # Скорость звука в модели


def InitBarrier():
    """ Создаем барьер и отталкивающие границы барьера """
    barrier0 = BarrierShape()

    barrier1 = np.roll(barrier0,  1, axis=1)
    barrier2 = np.roll(barrier1,  1, axis=0)
    barrier3 = np.roll(barrier0,  1, axis=0)
    barrier4 = np.roll(barrier0, -1, axis=1)
    barrier5 = np.roll(barrier4, -1, axis=0)
    barrier6 = np.roll(barrier0, -1, axis=0)

    return np.array([
        barrier0, 
        barrier1, barrier2, barrier3,  barrier4, barrier5, barrier6
    ])


def F_stat(Ux, Uy, Rho):
    """ Вычисляем статистическое распределение частиц в зависимости от общей скорости и плотности """
    UV = np.zeros((Q, Height, Width)) 
    for q in range(Q):
        UV[q] = (V[q,0]*Ux + V[q,1]*Uy)/C**2

    U2 = (Ux**2 + Uy**2)/C**2

    f_stat = np.zeros((Q, Height, Width))
    for q in range(Q):
        f_stat[q] = Rho * W[q] * (1 + UV[q] + UV[q]**2/2 - U2/2)

    return f_stat

def Mode0(f):
    """ Плотность """
    mode = np.zeros((Height, Width))
    for q in range(Q):
        mode += f[q]
    return mode

def Mode1(f):
    """ Плотность*Скорость """
    mode = np.zeros((D, Height, Width))
    for q in range(Q):
        for d1 in range(D):
            mode[d1] += f[q]*V[q,d1]
    return mode

def Mode2(f):
    """ Плотность*[Скорость x Скорость] минус тензор напряжения """
    mode = np.zeros((D,D,Height, Width))
    for q in range(Q):
        for d1 in range(D):
            for d2 in range(D):
                mode[d1,d2] += f[q]*V[q,d1]*V[q,d2]
    return mode


def iter(f, f_out, barrier):
    """stream"""
    (f0, f1, f2, f3, f4, f5, f6) = f

    f3[1:]  = f3[:-1]
    f2[1:] = f2[:-1]

    f6[:-1]  = f6[1:]
    f5[:-1] = f5[1:]

    f1[:,1:]  = f1[:,:-1]
    f2[:,1:] = f2[:,:-1]

    f4[:,:-1]  = f4[:,1:]
    f5[:,:-1] = f5[:,1:]

    """ BC_barrier """
    (b0, b1, b2, b3, b4, b5, b6) = barrier

    f1[b1]   = f4[b0]
    f2[b2]   = f5[b0]
    f3[b3]   = f6[b0]
    f4[b4]   = f1[b0]
    f5[b5]   = f2[b0]
    f6[b6]   = f3[b0]

    """ Calc U, Rho """
    Rho = Mode0(f)

    Ux, Uy = Mode1(f)
    Ux /= Rho
    Uy /= Rho

    """ Collide """

    f += (F_stat(Ux, Uy, Rho)-f)/(0.5 + Viscosity/C**2)

    """ BC_out """

    f[:,0,:] = f_out[:,0,:]
    f[:,-1,:] = f_out[:,-1,:]
    f[:,:,0] = f_out[:,:,0]
    f[:,:,-1] = f_out[:,:,-1]


def curl(ux, uy):
    """ двумерный ротор макроскопического поля скорости """ 
    return np.roll(uy,-1,axis=1) - np.roll(uy,1,axis=1) - np.roll(ux,-1,axis=0) + np.roll(ux,1,axis=0)

def main():

    Ux  = np.zeros((Height, Width)) + U0[0]
    Uy  = np.zeros((Height, Width)) + U0[1]
    Rho = np.ones((Height, Width))


    barrier = InitBarrier()

    F = F_stat(Ux, Uy, Rho)

    F_out = F_stat(Ux, Uy, Rho)

    for _ in range(1000):
        iter(F, F_out, barrier)


    
    Rho = Mode0(F)
    Ux, Uy = Mode1(F)

    # E = Mode2(F)
    Ux /= Rho
    Uy /= Rho

    fig, ax = plt.subplots()

    fluidImage = ax.imshow(curl(Ux, Uy), origin='lower', norm=plt.Normalize(-.1,.1), 
                                        cmap=plt.get_cmap('jet'), interpolation='none')

    plt.show()




if __name__ == '__main__':
    main()

