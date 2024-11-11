import matplotlib.pyplot, matplotlib.animation
import numpy as np


# Define constants:
height = 80                            # размеры решетки
width = 200
viscosity = 0.02                    # вязкость жидкости
omega = 1 / (3*viscosity + 0.5)        # параметр "релаксации"
u_0 = np.array([0.1, 0])                            # начальная и входящая скорость

u0 = 0.1

four9ths = 4.0/9.0                    # abbreviations for lattice-Boltzmann weight factors
one9th   = 1.0/9.0
one36th  = 1.0/36.0

density = 1

f = np.ones((9, height, width))

dir_template = np.array([
    [-1,1],[0,1],[1,1],
    [-1,0],[0,0],[1,0],
    [-1,-1],[0,-1],[1,-1]
])

coef_template = np.array([
    1/36,1/9,1/36,
    1/9, 4/9, 1/9,
    1/36,1/9,1/36
]).reshape(9,1,1)

u_dir_temp = (dir_template@u_0).reshape(9,1,1)

f = density * coef_template*(f + 3*u_dir_temp + 4.5*u_dir_temp@u_dir_temp - 1.5*u_0@u_0)

# Инициализируем все массивы для обеспечения равномерного правого потока

(fNW, fN, fNE, fW, f0, fE, fSW, fS, fSE) = f

Rho = f.sum(axis=0) 
dd = dir_template.T[0]

U_ = (f.T@dir_template).T/Rho
(U_x, U_y) = U_

# U_x = f_

Ux = (fE + fNE + fSE - fW - fNW - fSW) / Rho                # macroscopic x velocity
Uy = (fN + fNE + fNW - fS - fSE - fSW) / Rho                # macroscopic y velocity

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
def stream():
    global fN, fS, fE, fW, fNE, fNW, fSE, fSW

    f = np.array([fNW, fN, fNE, fW, f0, fE, fSW, fS, fSE])

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
        
# Collide particles within each cell to redistribute velocities (could be optimized a little more):
def collide():
    global Rho, Ux, Uy, f0, fN, fS, fE, fW, fNE, fNW, fSE, fSW

    f = np.array([fNW, fN, fNE, fW, f0, fE, fSW, fS, fSE])

    Rho = f.sum(axis=0) 
    (Ux, Uy) = (f.T@dir_template).T/Rho

    ux2 = Ux * Ux                # pre-compute terms used repeatedly...
    uy2 = Uy * Uy
    u2 = ux2 + uy2
    omu215 = 1 - 1.5*u2            # "one minus u2 times 1.5"
    uxuy = Ux * Uy
    f0 = (1-omega)*f0 + omega * four9ths * Rho * omu215
    fN = (1-omega)*fN + omega * one9th * Rho * (omu215 + 3*Uy + 4.5*uy2)
    fS = (1-omega)*fS + omega * one9th * Rho * (omu215 - 3*Uy + 4.5*uy2)
    fE = (1-omega)*fE + omega * one9th * Rho * (omu215 + 3*Ux + 4.5*ux2)
    fW = (1-omega)*fW + omega * one9th * Rho * (omu215 - 3*Ux + 4.5*ux2)
    fNE = (1-omega)*fNE + omega * one36th * Rho * (omu215 + 3*(Ux+Uy) + 4.5*(u2+2*uxuy))
    fNW = (1-omega)*fNW + omega * one36th * Rho * (omu215 + 3*(-Ux+Uy) + 4.5*(u2-2*uxuy))
    fSE = (1-omega)*fSE + omega * one36th * Rho * (omu215 + 3*(Ux-Uy) + 4.5*(u2-2*uxuy))
    fSW = (1-omega)*fSW + omega * one36th * Rho * (omu215 + 3*(-Ux-Uy) + 4.5*(u2+2*uxuy))
    # Force steady rightward flow at ends (no need to set 0, N, and S components):
    fE[:,0] = one9th * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    fW[:,0] = one9th * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    fNE[:,0] = one36th * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    fSE[:,0] = one36th * (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    fNW[:,0] = one36th * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    fSW[:,0] = one36th * (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)

# Compute curl of the macroscopic velocity field:
def curl(ux, uy):
    return np.roll(uy,-1,axis=1) - np.roll(uy,1,axis=1) - np.roll(ux,-1,axis=0) + np.roll(ux,1,axis=0)


# Here comes the graphics and animation...
theFig = matplotlib.pyplot.figure(figsize=(8,3))
fluidImage = matplotlib.pyplot.imshow(curl(Ux, Uy), origin='lower', norm=matplotlib.pyplot.Normalize(-.1,.1), 
                                    cmap=matplotlib.pyplot.get_cmap('jet'), interpolation='none')
        # See http://www.loria.fr/~rougier/teaching/matplotlib/#colormaps for other cmap options
bImageArray = np.zeros((height, width, 4), np.uint8)    # an RGBA image
bImageArray[barrier,3] = 255                                # set alpha=255 only at barrier sites
barrierImage = matplotlib.pyplot.imshow(bImageArray, origin='lower', interpolation='none')


def nextFrame(arg):                            # (arg is the frame number, which we don't need)
    stream()
    collide()
    fluidImage.set_array(curl(Ux, Uy))
    return (fluidImage, barrierImage)        # return the figure elements to redraw

animate = matplotlib.animation.FuncAnimation(theFig, nextFrame, interval=1, blit=True)
matplotlib.pyplot.show()