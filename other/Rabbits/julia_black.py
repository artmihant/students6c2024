# библиотеки
# инициализиация
import math

from PIL import Image
import numpy as np


def colorize(v, m):
    """ Принимает число (v) в интервале от 0 до m и возвращает цвет в виде числового кода rgba.
        Чем меньше value, тем светлее цвет (до белого) """

    main_color = np.array([0, 0, 0], dtype=np.float64)  # основной цвет
    shade_color = np.array([0, 0, 200], dtype=np.float64)  # дополнительный цвет

    h = 255

    rgb = np.array([
        main_color / h,
        shade_color / h,
        (np.array([h, h, h], dtype=np.float64) - main_color - shade_color)/h
    ])

    vector = np.array([1, (m - v) / m, (m - v) ** 2 / m ** 2])

    return rgb.transpose().dot(vector)*255


if __name__ == '__main__':

    # пусть c = p + iq и p меняется в диапазоне от pmin до pmax,
    # а q меняется в диапазоне от qmin до qmax
    pmin, pmax, qmin, qmax = -2, 2, -2, 2

    # число точек по горизонтали и вертикали
    ppoints, qpoints = 500, 500
    # максимальное количество итераций
    max_iterations = 25

    # если ушли на это расстояние, считаем, что ушли на бесконечность
    infinity_border = 10

    fl = 100

    frames = []
    for t in range(0, fl):

        image = Image.new('RGB', (ppoints, qpoints), 'black')
        canvas = np.array(image)
        
        p, q = np.mgrid[pmin:pmax:(ppoints * 1j), qmin:qmax:(qpoints * 1j)]


        alpha = 2*t*math.pi/fl
        c = 1*math.cos(alpha) + 1j*math.sin(alpha)

        z = p + 1j*q
        value = np.zeros((ppoints, qpoints))

        for k in range(max_iterations):
            z = z**2 + c

            mask = (np.abs(z) > infinity_border) & (value == np.array(0))

            canvas[mask] = np.array([*colorize(k, max_iterations)])

            z[mask] = np.nan

        image = Image.fromarray(canvas)
        frames.append(image)


    frames[0].save(
        'julia.gif',
        save_all=True,
        append_images=frames[1:],  # Срез который игнорирует первый кадр.
        optimize=True,
        duration=100,
        loop=1
    )
