import time
from random import random
from typing import Tuple
import sys

# area circulo / area cuadrado = nº puntos en circulo / nº puntos en cuadrado
# pi * r^2 / 4r^2 = nº puntos en circulo / nº puntos en cuadrado
# pi = nº puntos en circulo * 4 / nº puntos en cuadrado

# Cuadrado de 1x1

def ratio(point: Tuple[float, float]) -> float:
    return point[0]**2 + point[1]**2

def get_pi(n=1000, iter=10) -> float:

    pis = []
    for _ in range(iter):
        points_in_square = [(random(), random()) for _ in range(n)]
        points_in_circle = [point for point in points_in_square if ratio(point) < 1]
        pis.append(4 * len(points_in_circle) / len(points_in_square))
    
    return sum(pis) / len(pis)


if __name__ == "__main__":
        
    
    if len(sys.argv) == 2:
        n = int(sys.argv[1])
        iter = 1
    elif len(sys.argv) == 3:
        n = int(sys.argv[1])
        iter = int(sys.argv[2])
    else:
        n = 1000
        iter = 1
    t0 = time.time()
    pi = get_pi(n=n, iter=iter)
    total_time = time.time() - t0
    vel = n / total_time

    print(f"{n},{pi},{total_time},{vel}")
