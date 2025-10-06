#!/bin/bash

# Archivo CSV de salida
output="resultados.csv"

# Escribe la cabecera
echo "n,r,resultado" > "$output"

# Bucle doble
for n in 5000000 10000000 50000000; do
    echo "Ejecutando con n=$n y iter=1..."
    
    # Ejecutar el script de Python y capturar su salida
    result=$(python3 pi_montecarlo.py "$n" "1")
    
    # Añadir una línea al CSV
    echo "$n,$result" >> "$output"
done

echo "Ejecuciones completadas. Resultados guardados en $output"
