years = 20

def process(r, a):
    return a * r * (1 - r)

r = 0.01

a = 1.5

for y in range(years):
    print(f'{y}: {r}')
    r = process(r, a)

