#!/bin/bash

# Lista de pares de parámetros
for n in 5000000 10000000 50000000; do
    python pi_montecarlo.py $n $r
done