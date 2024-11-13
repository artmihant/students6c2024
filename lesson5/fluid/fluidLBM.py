import time
import matplotlib.pyplot as plt, matplotlib.animation
import numpy as np
import numba as nb

""" Зададим константы и свойства задачи """

Viscosity = 0.01                                # вязкость жидкости
Height, Width = 80, 200,                          # размеры решетки

U0 = np.array([0.15, 0])                        # начальная и внешняя скорость (в махах)

Ux  = np.zeros((Height, Width)) + U0[0]
Uy  = np.zeros((Height, Width)) + U0[1]
Rho = np.ones((Height, Width))

Ux0  = np.zeros((Height, Width)) + U0[0]
Uy0  = np.zeros((Height, Width)) + U0[1]
Rho0 = np.ones((Height, Width))

def BarrierShape():
    """ Инициализируем форму барьера. Необходимо установить True там, где барьер """
    barrier = np.zeros((Height, Width), bool)

    # круг
    for i in range(barrier.shape[0]):
        for j in range(barrier.shape[1]):
            if (i - Height//2)**2 + (j - Height//2)**2 < (Height//10)**2:
                barrier[i,j] = True
    # хвост
    barrier[(Height//2), ((Height//2)):((Height//2)+4*(Height//10))] = True

    return barrier


""" Зададим свойства шаблона решетки D2Q9 """

D = 2 # Мерность модели
Q = 9 # число точек в шаблоне

V = np.array([
    [-1, 1],[ 0, 1],[ 1, 1],
    [-1, 0],[ 0, 0],[ 1, 0],
    [-1,-1],[ 0,-1],[ 1,-1]
])

W = np.array([
    1/36, 1/9, 1/36,
    1/9,  4/9, 1/9,
    1/36, 1/9, 1/36
])

C = 1/3**0.5  # Скорость звука в модели


def InitBarrier():
    """ Создаем барьер и отталкивающие границы барьера """
    barrierC = BarrierShape()

    barrierN = np.roll(barrierC,  1, axis=0)
    barrierS = np.roll(barrierC, -1, axis=0)
    barrierE = np.roll(barrierC,  1, axis=1)
    barrierW = np.roll(barrierC, -1, axis=1)
    barrierNE = np.roll(barrierN,  1, axis=1)
    barrierNW = np.roll(barrierN, -1, axis=1)
    barrierSE = np.roll(barrierS,  1, axis=1)
    barrierSW = np.roll(barrierS, -1, axis=1)

    return np.array([
        barrierNW, barrierN, barrierNE,
        barrierW,  barrierC, barrierE,
        barrierSW, barrierS, barrierSE
    ])


def F_stat(Ux, Uy, Rho):
    """ Вычисляем статистическое распределение частиц в зависимости от общей скорости и плотности """
    UV = np.zeros((9, Height, Width)) 
    for i in range(9):
        UV[i] = (V[i,0]*Ux + V[i,1]*Uy)/C**2

    U2 = (Ux**2 + Uy**2)/C**2

    f_stat = np.zeros((9, Height, Width))
    for i in range(9):
        f_stat[i] = Rho * W[i] * (1 + UV[i] + UV[i]**2/2 - U2/2)

    return f_stat

def Mode0(f):
    """ Плотность """
    mode = np.zeros((Height, Width))
    for i in range(9):
        mode += f[i]
    return mode

def Mode1(f):
    """ Плотность*Скорость """
    mode = np.zeros((2, Height, Width))
    for i in range(9):
        for d1 in range(2):
            mode[d1] += f[i]*V[i,d1]
    return mode

def Mode2(f):
    """ Плотность*[Скорость x Скорость] минус тензор напряжения """
    mode = np.zeros((2,2,Height, Width))
    for i in range(9):
        for d1 in range(2):
            for d2 in range(2):
                mode[d1,d2] += f[i]*V[i,d1]*V[i,d2]
    return mode


def iter(f, f_out, barrier):
    """stream"""
    (fNW, fN, fNE, fW, fC, fE, fSW, fS, fSE) = f

    for y in range(Height-1,0,-1):
        fN[y]  = fN[y-1]
        fNE[y] = fNE[y-1]
        fNW[y] = fNW[y-1]

    for y in range(0,Height-1):
        fS[y]  = fS[y+1]
        fSE[y] = fSE[y+1]
        fSW[y] = fSW[y+1]

    for x in range(Width-1,0,-1):
        fE[:,x]  = fE[:,x-1]
        fNE[:,x] = fNE[:,x-1]
        fSE[:,x] = fSE[:,x-1]

    for x in range(0,Width-1):
        fW[:,x]  = fW[:,x+1]
        fNW[:,x] = fNW[:,x+1]
        fSW[:,x] = fSW[:,x+1]

    """ BC_barrier """
    (bNW, bN, bNE, bW, bC, bE, bSW, bS, bSE) = barrier

    fN[bN]   = fS[bC]
    fS[bS]   = fN[bC]
    fE[bE]   = fW[bC]
    fW[bW]   = fE[bC]
    fNE[bNE] = fSW[bC]
    fNW[bNW] = fSE[bC]
    fSE[bSE] = fNW[bC]
    fSW[bSW] = fNE[bC]

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

    barrier = InitBarrier()

    F = F_stat(Ux, Uy, Rho)

    F_out = F_stat(Ux, Uy, Rho)

    fig, ax = plt.subplots()

    fluidImage = ax.imshow(curl(Ux, Uy), origin='lower', norm=plt.Normalize(-.1,.1), 
                                        cmap=plt.get_cmap('jet'), interpolation='none')

    bImageArray = np.zeros((Height, Width, 4), np.uint8)
    bImageArray[barrier[4],3] = 100
    barrierImage = plt.imshow(bImageArray, origin='lower', interpolation='none')


    def nextFrame(_):

        for i in range(40):
            iter(F, F_out, barrier)

        Rho = Mode0(F)
        Ux, Uy = Mode1(F)

        E = Mode2(F)
        Ux /= Rho
        Uy /= Rho

        fluidImage.set_array(curl(Ux, Uy))
        return (fluidImage, barrierImage)        

    animate = matplotlib.animation.FuncAnimation(fig, nextFrame, interval=1, blit=True)
    plt.show()



if __name__ == '__main__':
    main()

