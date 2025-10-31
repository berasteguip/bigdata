"""
Ejecucion con bucles for
"""
import pandas as pd
from dask.distributed import Client, wait
import time
import random
import math

# Conexion al scheduler de Dask (ajusta la URL si es diferente)
client = Client("tcp://localhost:8786")

def montecarlo(num_puntos, seed):
    puntos_dentro = 0
    random.seed(seed)
    for _ in range(num_puntos):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            puntos_dentro += 1
    # Devolvemos el numero de aciertos y el total de puntos generados
    return puntos_dentro, num_puntos

num_puntos_totales = [5 * 10**6, 10 * 10**6, 50 * 10**6]
all_rows = []

for num_total in num_puntos_totales:
    t0 = time.time()

    # Dividimos el trabajo en 200 tareas
    chunk = num_total // 200
    futures = [client.submit(montecarlo, chunk, i) for i in range(200)]

    # Esperamos a que terminen todas las tareas
    wait(futures)
    results = client.gather(futures)

    # Sumamos los resultados
    total_dentro = sum(r[0] for r in results)
    total_puntos = sum(r[1] for r in results)

    # Calculo final de pi
    pi_est = 4 * total_dentro / total_puntos

    t_proceso = time.time() - t0
    vel = num_total / t_proceso


    print(f"\nNumero total de puntos: {num_total}")
    print(f"Estimacion de pi: {pi_est:.6f}")
    print(f"Tiempo de proceso: {t_proceso:.2f} s")
    all_rows.append({
            "n": num_total,
            "pi": pi_est,
            "time": t_proceso,
            "vel": vel,
            "precision": abs((pi_est - math.pi) / math.pi)
        })
df = pd.DataFrame(all_rows)
df.to_csv('resultados_dask_for.csv', index=False)
print('Results saved to resultados_dask_for.csv')