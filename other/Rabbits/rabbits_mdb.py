import numpy as np
# import numexpr as ne
import matplotlib.pyplot as plt
import random
from multiprocessing import Pool

def process(r, a):
    # return r**2 - a
    # return  1j*a*np.exp(r)
    # if a == 1j:
    #     return 0
    return a * r * (1-r)

    # if a == 1:
    #     return 0
    # return a * r * (1-r)
 

def calc(a):

    r = 0.5
    
    seq = []

    for k in range(128):
        r = process(r, a)

        if abs(r) > 1:
            return 256-k

        for s in seq:
            if abs(s-r) < 0.001:
                return k

        seq.append(r)

    return 128

    # for k in range(1024):
    #     s = r
    #     r = process(r, a)
    #     if abs(s-r) < 0.0001:
    #         return 0
    #     if abs(r) > 1:
    #         return 1

    # point = r
    # for k in range(64):
    #     r = process(r, a)
    #     if abs(r) > 1:
    #         return 1
            
    #     if abs(point - r) < 0.001:
    #         return 0
            
    # return 0



def draw_mandelbrot_rabbits_chart(a_range, dpu):

    width  = a_range[0][1]-a_range[0][0]
    height = a_range[1][1]-a_range[1][0]

    if width < height:
        x_points, y_points = int(dpu*width/height), dpu
    else:
        x_points, y_points = dpu, int(dpu*height/width)

    xx = np.linspace(*a_range[0], x_points).reshape(1, -1)
    yy = np.linspace(*a_range[1], y_points)[::-1].reshape(-1, 1)
    zz = xx + 1j * yy
 
    zz = zz.ravel()

    # zz = (zz-3j)/(zz-1j)

    p = Pool(6)

    # image = ne.evaluate('abs(zz)') 

    image = p.map(calc, zz)
    
    # image = np.vectorize(calc)(zz)


    image = np.reshape(image, (y_points, x_points))

    plt.xticks([])
    plt.yticks([])

    # cmap in ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    plt.imshow(image, cmap="inferno")
    plt.savefig(f'./test.png', pad_inches=0, dpi=300)
    plt.show()


if __name__ == '__main__':


    dpu = 300

    # пусть a = x + iy и x меняется в диапазоне от 0 до 4; y - от -2 до 2
    # a_range = [3.735, 3.745]

    a_range = [[-2, 4], [-2, 2]]


    import time
    start_time = time.time()
    # for years in range(100):
    draw_mandelbrot_rabbits_chart(a_range, dpu)

    print("--- %s seconds ---" % (time.time() - start_time))