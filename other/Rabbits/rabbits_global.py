import random
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation
from numpy import vectorize


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

    return *rgb.transpose().dot(vector), 1


def draw_global_rabbits_chart(a_range, r_range, years, fps, r0, process):
    """ Рисует динамический график числа кроликов (r) от заданного параметра (а) """

    # число вариантов а в заданном интервале
    values_count = 1000

    # число кадров, остающихся на экране
    max_visible_frames = 12

    process = vectorize(process)
    r0 = vectorize(r0)

    a_values = np.linspace(*a_range, values_count)

    rabbits = r0(a_values)

    fig, axes = plt.subplots()
    axes.set_xlim(*a_range)
    axes.set_ylim(*r_range)

    axes.set_xlabel('a')
    axes.set_ylabel('rabbits')
    plt.title(f'{years} лет кролиководства за {years//fps} секунд')

    frames = []

    last_rabbits = []

    for _ in range(years):
        rabbits = process(rabbits, a_values)
        last_rabbits.append(rabbits)
        if len(last_rabbits) > max_visible_frames:
            last_rabbits = last_rabbits[1:]

        frame = []
        for index, rabbit in enumerate(last_rabbits):
            line, = axes.plot(a_values, rabbit, color=colorize(index, len(last_rabbits)), linewidth='0.5')

            frame.append(line)
        frames.append(frame)

    animation = ArtistAnimation(
        fig,
        frames,
        interval=1000 // fps,
        blit=True,
        repeat=True)

    plt.show()


if __name__ == '__main__':
    # общее число лет, пока мы выращиваем кроликов
    years = 300

    # число кадров в секунду (один кадр - один год)
    fps = 20

    """Кроличья популяция по закону a*r*(1-r)"""

    # интервалы для a и r - в каких пределах они определены
    a_range = [-2, 4]
    r_range = [0, 1]

    # функция инициализирующего значения для r
    def r0(a):
        return 0.5

    # итерационный процесс
    def process(r, a):
        return a * r * (1 - r)

    """Кроличья популяция по закону a * math.sin(math.pi/2*r)"""

    # a_range = [0, 2]
    # r_range = [0, 2]

    # def r0(a):
    #     return 1

    # def process(r, a):
    #     return a * math.sin(math.pi/2*r)

    """ Кроличья популяция по закону r**2 - a """
    #
    # a_range = [0, 2]
    # r_range = [2, -2]
    #
    # def r0(a):
    #     return 0
    #
    # def process(r, a):
    #     if r > 1000:
    #         return 1000
    #     if r < -1000:
    #         return -1000
    #     return r**2 - a

    draw_global_rabbits_chart(a_range, r_range, years, fps, r0, process)
