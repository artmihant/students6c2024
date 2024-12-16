import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

width, height = 300, 300
max_iter = 200

x_min, x_max = -2, 2
y_min, y_max = (x_min-x_max)/2, (x_max-x_min)/2

zstart = complex(-1, -1)
zend = complex(1,1)

def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, z0):
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    real, imag = np.meshgrid(real, imag)
    c = real + 1j * imag

    z = np.full(c.shape, z0, dtype=np.complex128)
    divergence_step = np.full(c.shape, max_iter, dtype=int)

    for i in range(max_iter):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] ** 2 + c[mask]

        diverged = (np.abs(z) > 2) & (divergence_step == max_iter)
        divergence_step[diverged] = i

    return divergence_step

#plt.imshow(image, cmap='copper', extent=(x_min, x_max, y_min, y_max))
#plt.colorbar(label="Iters before devergence")
#plt.title("Mandelbrot fractal")
#plt.xlabel("Re")
#plt.ylabel("Im")
#plt.show()

image = mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, complex(0,0))
Fig, Ax = plt.subplots()
x_grid, y_grid = np.meshgrid(np.linspace(x_min,x_max,width), np.linspace(x_min,x_max,height))
graph = Ax.pcolormesh(x_grid,y_grid,image,cmap='RdPu')
Ax.set_xlabel("Re")
Ax.set_ylabel("Im")
Ax.legend()

dz = (zend - zstart) / 100
print("dz: ",dz)
def loop_animation(i):
    curz = zstart + dz * i
    image = mandelbrot(x_min, x_max, y_min, y_max, width, height, max_iter, curz)
    Ax.set_title(curz)
    graph.set_array(image)
    return graph

ani = animation.FuncAnimation(
    fig=Fig, 
    func=loop_animation, 
    frames=100, 
    interval=1000/20
)
ani.save("mandelbrot-1-1to11.mkv", writer='ffmpeg')
#plt.show()