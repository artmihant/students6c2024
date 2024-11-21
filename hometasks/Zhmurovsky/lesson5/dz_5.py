import time
import matplotlib.pyplot as plt, matplotlib.animation
import numpy as np
import numba as nb

""" Зададим константы и свойства задачи """

Viscosity = 0.01  # вязкость жидкости
Height, Width = 80, 200  # размеры решетки

U0 = np.array([0.15, 0])  # начальная и внешняя скорость (в махах)

Ux = np.zeros((Height, Width)) + U0[0]
Uy = np.zeros((Height, Width)) + U0[1]
Rho = np.ones((Height, Width))

Ux0 = np.zeros((Height, Width)) + U0[0]
Uy0 = np.zeros((Height, Width)) + U0[1]
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

D = 2  # Мерность модели
Q = 9  # число точек в шаблоне

V = np.array([
    [-1, 1], [0, 1], [1, 1],
    [-1, 0], [0, 0], [1, 0],
    [-1, -1], [0, -1], [1, -1]
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


@nb.njit()
def F_stat(Ux, Uy, Rho):
    """ Вычисляем статистическое распределение частиц в зависимости от общей скорости и плотности """
    UV = np.zeros((9, Height, Width)) 
    for i in range(9):
        UV[i] = (V[i, 0]*Ux + V[i, 1]*Uy) / C**2

    U2 = (Ux**2 + Uy**2) / C**2

    f_stat = np.zeros((9, Height, Width))
    for i in range(9):
        f_stat[i] = Rho * W[i] * (1 + UV[i] + UV[i]**2 / 2 - U2 / 2)

    return f_stat

@nb.njit()
def Mode0(f):
    """ Плотность """
    mode = np.zeros((Height, Width))
    for i in range(9):
        mode += f[i]
    return mode


@nb.njit()
def Mode1(f):
    """ Плотность*Скорость """
    mode = np.zeros((2, Height, Width))
    for i in range(9):
        for d1 in range(2):
            mode[d1] += f[i]*V[i, d1]
    return mode


@nb.njit()
def Mode2(f):
    """ Плотность*[Скорость x Скорость] минус тензор напряжения """
    mode = np.zeros((2, 2, Height, Width))
    for i in range(9):
        for d1 in range(2):
            for d2 in range(2):
                mode[d1, d2] += f[i] * V[i, d1] * V[i, d2]
    return mode

@nb.njit()
def iter(f, f_out, barrier):
    
    fNW, fN, fNE, fW, fC, fE, fSW, fS, fSE = f

    fNW[1:Height] = fNW[0:Height-1]
    fN[1:Height] = fN[0:Height-1]
    fNE[1:Height] = fNE[0:Height-1]

    fS[0:Height-1] = fS[1:Height]
    fSE[0:Height-1] = fSE[1:Height]
    fSW[0:Height-1] = fSW[1:Height]

    fE[:, 1:Width] = fE[:, 0:Width-1]
    fNE[:, 1:Width] = fNE[:, 0:Width-1]
    fSE[:, 1:Width] = fSE[:, 0:Width-1]

    fW[:, 0:Width-1] = fW[:, 1:Width]
    fNW[:, 0:Width-1] = fNW[:, 1:Width]
    fSW[:, 0:Width-1] = fSW[:, 1:Width]

    barrierN, barrierS, barrierE, barrierW, barrierC, barrierNE, barrierNW, barrierSE, barrierSW = barrier

    for y in range(Height):
        for x in range(Width):
            if barrierN[y, x]:
                fN[y, x] = fS[y, x]
            if barrierS[y, x]:
                fS[y, x] = fN[y, x]
            if barrierE[y, x]:
                fE[y, x] = fW[y, x]
            if barrierW[y, x]:
                fW[y, x] = fE[y, x]
            if barrierNE[y, x]:
                fNE[y, x] = fSW[y, x]
            if barrierNW[y, x]:
                fNW[y, x] = fSE[y, x]
            if barrierSE[y, x]:
                fSE[y, x] = fNW[y, x]
            if barrierSW[y, x]:
                fSW[y, x] = fNE[y, x]
                
    Rho = Mode0(f)

    Ux, Uy = Mode1(f)
    Ux /= Rho
    Uy /= Rho

    f += (F_stat(Ux, Uy, Rho) - f) / (0.5 + Viscosity / C ** 2)

    f[:, 0, :] = f_out[:, 0, :]
    f[:, -1, :] = f_out[:, -1, :]
    f[:, :, 0] = f_out[:, :, 0]
    f[:, :, -1] = f_out[:, :, -1]





@nb.njit()
def curl(ux, uy):
    """ двумерный ротор макроскопического поля скорости """
    curl_result = np.zeros_like(ux)
    
    # Для сдвигов по оси X (по горизонтали)
    curl_result[:, 1:-1] = uy[:, 2:] - uy[:, :-2]
    
    # Для сдвигов по оси Y (по вертикали)
    curl_result[1:-1, :] -= ux[2:, :] - ux[:-2, :]

    return curl_result


def main():

    barrier = InitBarrier()

    F = F_stat(Ux, Uy, Rho)

    F_out = F_stat(Ux, Uy, Rho)

    fig, ax = plt.subplots()

    fluidImage = ax.imshow(curl(Ux, Uy), origin='lower', norm=plt.Normalize(-.1, .1),
                           cmap=plt.get_cmap('jet'), interpolation='none')

    bImageArray = np.zeros((Height, Width, 4), np.uint8)
    bImageArray[barrier[4], 3] = 100
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
    animate = matplotlib.animation.FuncAnimation(
        fig,
        nextFrame,
        frames=100,
        interval=1,
        blit=True,
        cache_frame_data=False
    )
    animate.save(filename="simulation.gif", writer="imagemagick")
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
 