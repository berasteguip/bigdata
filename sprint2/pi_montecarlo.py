import time
import pandas as pd
from random import random, Random
from typing import Tuple, List
import math
import numpy as np
from dask.distributed import Client

# Conéctate a tu scheduler Dask (o deja vacío para local)
client = Client(address="tcp://localhost:8786")  # o Client() para local

# area circulo / area cuadrado = nº puntos en circulo / nº puntos en cuadrado
# pi * r^2 / 4r^2 = nº puntos en circulo / nº puntos en cuadrado
# pi = nº puntos en circulo * 4 / nº puntos en cuadrado

def ratio(point: Tuple[float, float]) -> float:
    return point[0]**2 + point[1]**2

N_TASKS = 200

def point_in_circle(n: int) -> int:

    np.random.seed()  # Asegura diferentes semillas en cada worker

    x = np.random.rand(n)
    y = np.random.rand(n)
    inside = int(np.sum(x**2 + y**2 <= 1))
    return inside
    
def pi_montecarlo(n) -> float:
    per_task = n // N_TASKS
    futures = [client.submit(point_in_circle, per_task) for _ in range(N_TASKS)]
    results = client.gather(futures)
    total_points_in_circle = sum(results)
    pi = 4 * total_points_in_circle / (per_task * N_TASKS)
    return pi

def execution(params: List[int]) -> None:
    print("Dask Monte Carlo Pi Estimation")
    results = []
    for n in params:
        t0 = time.time()
        pi = pi_montecarlo(n)
        total_time = time.time() - t0
        vel = n / total_time
        results.append({
            "n": n,
            "pi": pi,
            "time": total_time,
            "vel": vel,
            "precision": abs((pi - math.pi) / math.pi)
        })
    df = pd.DataFrame(results)
    df.to_csv('resultados_dask.csv', index=False)
    print('Results saved to resultados_dask.csv')

if __name__ == "__main__":
    params = [5*10**6, 10*10**6, 50*10**6]
    execution(params)