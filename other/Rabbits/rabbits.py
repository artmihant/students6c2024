import time

def rabbits(r0, a, years, process):
    r = r0
    for i in range(years):
        print(r)
        r = process(r, a)
        time.sleep(1)


if __name__ == '__main__':
    # общее число лет, пока мы выращиваем кроликов
    years = 100

    # итерационный процесс
    def process(r, a):
        return a * r * (1 - r)

    r0 = 0.3

    a = 2

    rabbits(r0, a, years, process)