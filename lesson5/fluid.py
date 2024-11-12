import matplotlib.pyplot, matplotlib.animation
import numpy as np


# Define constants:
height = 80                          # размеры решетки
width = 200
viscosity = 0.02                    # вязкость жидкости
omega = 1 / (3*viscosity + 0.5)        # параметр "релаксации"
u_0 = np.array([0.1, 0])                            # начальная и входящая скорость

u0 = 0.1

four9ths = 4.0/9.0                    # abbreviations for lattice-Boltzmann weight factors
one9th   = 1.0/9.0
one36th  = 1.0/36.0


f = np.ones((9, height, width))

DirTemp = np.array([
    [-1,1],[0,1],[1,1],
    [-1,0],[0,0],[1,0],
    [-1,-1],[0,-1],[1,-1]
]).reshape((9,2,1,1))

CoefTemp = np.array([
    1/36,1/9,1/36,
    1/9, 4/9, 1/9,
    1/36,1/9,1/36
]).reshape((9,1,1))

U0_x = 0.1
U0_y = 0

Ux = np.zeros((1, height, width)) + U0_x
Uy = np.zeros((1, height, width)) + U0_y

Rho = np.ones((1, height, width))

U_dir = DirTemp[:,0]*Ux + DirTemp[:,1]*Uy
U2 = (Ux**2 + Uy**2)

f = Rho * CoefTemp * (1 + 3*U_dir + 4.5*U_dir**2 - 1.5*U2)

# Инициализируем все массивы для обеспечения равномерного правого потока


# Initialize barriers:
barrier = np.zeros((height,width), bool)                    # True wherever there's a barrier
barrier[((height//2)-8):((height//2)+8), height//2] = True            # simple linear barrier
barrierN = np.roll(barrier,  1, axis=0)                    # sites just north of barriers
barrierS = np.roll(barrier, -1, axis=0)                    # sites just south of barriers
barrierE = np.roll(barrier,  1, axis=1)                    # etc.
barrierW = np.roll(barrier, -1, axis=1)
barrierNE = np.roll(barrierN,  1, axis=1)
barrierNW = np.roll(barrierN, -1, axis=1)
barrierSE = np.roll(barrierS,  1, axis=1)
barrierSW = np.roll(barrierS, -1, axis=1)


# Move all particles by one step along their directions of motion (pbc):
def stream(f):

    (fNW, fN, fNE, fW, f0, fE, fSW, fS, fSE) = f

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

    # Use tricky boolean arrays to handle barrier collisions (bounce-back):

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

# Collide particles within each cell to redistribute velocities (could be optimized a little more):
def collide(f):

    Rho = f.sum(axis=0).reshape((1, height, width))
    Ux = ((DirTemp[:,0]*f)).sum(axis=0).reshape((1, height, width))/Rho
    Uy = ((DirTemp[:,1]*f)).sum(axis=0).reshape((1, height, width))/Rho

    U_dir = DirTemp[:,0]*Ux + DirTemp[:,1]*Uy
    U2 = (Ux**2 + Uy**2)

    f = (1-omega)*f + omega * Rho * CoefTemp * (1 + 3*U_dir + 4.5*U_dir**2 - 1.5*U2)
 
    (fNW, fN, fNE, fW, f0, fE, fSW, fS, fSE) = f
 
    # Force steady rightward flow at ends (no need to set 0, N, and S components):

    U0_2 = (U0_x**2+U0_y**2)

    fE[:,0] = one9th * (1 + 3*U0_x + 4.5*U0_x**2 - 1.5*U0_2)
    fW[:,0] = one9th * (1 - 3*U0_x + 4.5*U0_x**2 - 1.5*U0_2)

    fN[0,:] = one9th * (1 + 3*U0_y + 4.5*U0_y**2 - 1.5*U0_2)
    fS[:,0] = one9th * (1 - 3*U0_y + 4.5*U0_y**2 - 1.5*U0_2)

    fNE[:,0] = one36th * (1 + 3*U0_x + 4.5*U0_x**2 - 1.5*U0_2)
    fSE[:,0] = one36th * (1 + 3*U0_x + 4.5*U0_x**2 - 1.5*U0_2)
    fNW[:,0] = one36th * (1 - 3*U0_x + 4.5*U0_x**2 - 1.5*U0_2)
    fSW[:,0] = one36th * (1 - 3*U0_x + 4.5*U0_x**2 - 1.5*U0_2)

    f[:] = np.array([fNW, fN, fNE, fW, f0, fE, fSW, fS, fSE])
    return f

# Compute curl of the macroscopic velocity field:
def curl(ux, uy):
    ux = ux.reshape((height, width))
    uy = uy.reshape((height, width))
    return np.roll(uy,-1,axis=1) - np.roll(uy,1,axis=1) - np.roll(ux,-1,axis=0) + np.roll(ux,1,axis=0)


# Here comes the graphics and animation...
theFig = matplotlib.pyplot.figure(figsize=(8,3))
fluidImage = matplotlib.pyplot.imshow(curl(Ux, Uy), origin='lower', norm=matplotlib.pyplot.Normalize(-.1,.1), 
                                    cmap=matplotlib.pyplot.get_cmap('jet'), interpolation='none')

bImageArray = np.zeros((height, width, 4), np.uint8)
bImageArray[barrier,3] = 255                              
barrierImage = matplotlib.pyplot.imshow(bImageArray, origin='lower', interpolation='none')


def nextFrame(arg):
    global f      

    for i in range(20):                  
        f = stream(f)
        f = collide(f)
    
    Rho = f.sum(axis=0)
    Ux = ((DirTemp[:,0]*f)).sum(axis=0)/Rho
    Uy = ((DirTemp[:,1]*f)).sum(axis=0)/Rho

    fluidImage.set_array(curl(Ux, Uy))
    return (fluidImage, barrierImage)        

animate = matplotlib.animation.FuncAnimation(theFig, nextFrame, interval=1, blit=True)
matplotlib.pyplot.show()