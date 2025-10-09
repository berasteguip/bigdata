import time
import pandas as pd
from random import random
from typing import Tuple, List
import sys
import math
import ray

ray.init("ray://localhost:10001")

# area circulo / area cuadrado = nº puntos en circulo / nº puntos en cuadrado
# pi * r^2 / 4r^2 = nº puntos en circulo / nº puntos en cuadrado
# pi = nº puntos en circulo * 4 / nº puntos en cuadrado

# Cuadrado de 1x1

def ratio(point: Tuple[float, float]) -> float:
    return point[0]**2 + point[1]**2

N_TASKS = 200

@ray.remote
def pi_montecarlo(n=1000, iter=10) -> float:

    pis = []
    for _ in range(iter):
        points_in_square = [(random(), random()) for _ in range(n)]
        points_in_circle = [point for point in points_in_square if ratio(point) < 1]
        pis.append(4 * len(points_in_circle) / len(points_in_square))
    
    return sum(pis) / len(pis)


def execution(params: List[int]) -> None:

    results = []
    for n in params:
        t0 = time.time()
        pi_futures = [pi_montecarlo.remote(n=n, iter=iter) for _ in range(N_TASKS)]
        pi_results = ray.get(pi_futures)
        total_time = time.time() - t0
        pi = sum(pi_results) / N_TASKS
        vel = n / total_time

        results.append({
            "n": n,
            "pi": pi,
            "time": total_time,
            "vel": vel,
            "precision": abs((pi - math.pi) / math.pi)
        })

    df = pd.DataFrame(results)
    
    df.to_csv(f'resultados_w1.csv')
    


if __name__ == "__main__":
        
    params = [5*10**6, 10*10**6, 50*10**6]
    execution(params)