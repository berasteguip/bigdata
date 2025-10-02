from random import random
from typing import Tuple

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

    print(get_pi(n=10**6, iter=5))
