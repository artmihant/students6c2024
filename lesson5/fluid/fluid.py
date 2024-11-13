import matplotlib.pyplot, matplotlib.animation
import numpy as np

# Зададим константы

Viscosity = 0.02                                # вязкость жидкости

Height = 80                                     # размеры решетки
Width = 200
Radius = Height//10

U0 = np.array([0.15, 0])                        # начальная и внешняя скорость (в махах)

Ux  = np.zeros((Height, Width)) + U0[0]
Uy  = np.zeros((Height, Width)) + U0[1]
Rho = np.ones((Height, Width))

# Зададим свойства шаблона решетки

V = np.array([
    [-1,1],[0,1],[1,1],
    [-1,0],[0,0],[1,0],
    [-1,-1],[0,-1],[1,-1]
]).reshape((9,2,1,1))


CoefTemp = np.array([
    1/36,1/9,1/36,
    1/9, 4/9, 1/9,
    1/36,1/9,1/36
]).reshape((9,1,1))

C = 1/3**0.5
Tau = Viscosity/C**2 + 0.5                     # параметр релаксации

def F_stat(Ux, Uy, Rho):
    UV = (V[:,0]*Ux + V[:,1]*Uy)/C**2
    U2 = (Ux**2 + Uy**2)/C**2
    return Rho * CoefTemp * (1 + UV + UV**2/2 - U2/2)

def U_Rho(f):
    Rho = f.sum(axis=0)
    Ux = ((V[:,0]*f)).sum(axis=0)/Rho
    Uy = ((V[:,1]*f)).sum(axis=0)/Rho
    return Ux, Uy, Rho  

F = F_stat(Ux, Uy, Rho)
F_0 = F_stat(Ux, Uy, Rho)


def InitBarrier():
    # Инициализируем форму барьера
    barrier = np.zeros((Height,Width), bool)                    

    # True там где барьер
    # круг
    for i in range(barrier.shape[0]):
        for j in range(barrier.shape[1]):
            if (i - Height//2)**2 + (j - Height//2)**2 < Radius**2:
                barrier[i,j] = True
    # хвост
    # barrier[(Height//2), ((Height//2)):((Height//2)+4*Radius)] = True            # simple linear barrier

    return barrier


barrier = InitBarrier()

barrierN = np.roll(barrier,  1, axis=0)                    
barrierS = np.roll(barrier, -1, axis=0)                    
barrierE = np.roll(barrier,  1, axis=1)                    
barrierW = np.roll(barrier, -1, axis=1)
barrierNE = np.roll(barrierN,  1, axis=1)
barrierNW = np.roll(barrierN, -1, axis=1)
barrierSE = np.roll(barrierS,  1, axis=1)
barrierSW = np.roll(barrierS, -1, axis=1)


# Переместить все частицы на один шаг вдоль направления их движения (pbc):
def stream(f):

    (fNW, fN, fNE, fW, f0, fE, fSW, fS, fSE) = f

    # fN[1:,:] = fN[:-1,:]

    fN  = np.roll(fN,   1, axis=0)    # axis 0 is north-south; + direction is north

    fNE = np.roll(fNE,  1, axis=0)
    fNW = np.roll(fNW,  1, axis=0)
    fS  = np.roll(fS,  -1, axis=0)

    fSE = np.roll(fSE, -1, axis=0)
    fSW = np.roll(fSW, -1, axis=0)

    fE  = np.roll(fE,   1, axis=1)    # axis 1 is east-west; + direction is east

    fNE = np.roll(fNE,  1, axis=1)
    fSE = np.roll(fSE,  1, axis=1)

    fW  = np.roll(fW,  -1, axis=1)

    fNW = np.roll(fNW, -1, axis=1)
    fSW = np.roll(fSW, -1, axis=1)


    fN[barrierN] = fS[barrier]
    fS[barrierS] = fN[barrier]
    fE[barrierE] = fW[barrier]
    fW[barrierW] = fE[barrier]
    fNE[barrierNE] = fSW[barrier]
    fNW[barrierNW] = fSE[barrier]
    fSE[barrierSE] = fNW[barrier]
    fSW[barrierSW] = fNE[barrier]
        
    f[:] = np.array([fNW, fN, fNE, fW, f0, fE, fSW, fS, fSE])
    return f

# Сталкиваем частицы внутри каждой ячейки, чтобы перераспределить скорости
def collide(f):

    Ux, Uy, Rho = U_Rho(f)

    f -= (f - F_stat(Ux, Uy, Rho))/Tau

    f[:,0,:] = F_0[:,0,:]
    f[:,-1,:] = F_0[:,-1,:]
    f[:,:,0] = F_0[:,:,0]
    f[:,:,-1] = F_0[:,:,-1]

    return f

# Вычислить ротор макроскопического поля скорости:
def curl(ux, uy):
    ux = ux.reshape((Height, Width))
    uy = uy.reshape((Height, Width))
    return np.roll(uy,-1,axis=1) - np.roll(uy,1,axis=1) - np.roll(ux,-1,axis=0) + np.roll(ux,1,axis=0)


theFig = matplotlib.pyplot.figure(figsize=(8,3))
fluidImage = matplotlib.pyplot.imshow(curl(Ux, Uy), origin='lower', norm=matplotlib.pyplot.Normalize(-.1,.1), 
                                    cmap=matplotlib.pyplot.get_cmap('jet'), interpolation='none')

bImageArray = np.zeros((Height, Width, 4), np.uint8)
bImageArray[barrier,3] = 255                              
barrierImage = matplotlib.pyplot.imshow(bImageArray, origin='lower', interpolation='none')


def nextFrame(arg):
    global F      

    for i in range(20):
        F = stream(F)
        F = collide(F)
    

    # print(F.min(), F.max())
    Ux, Uy, Rho =  U_Rho(F)

    fluidImage.set_array(curl(Ux, Uy))
    return (fluidImage, barrierImage)        

animate = matplotlib.animation.FuncAnimation(theFig, nextFrame, interval=1, blit=True)
matplotlib.pyplot.show()