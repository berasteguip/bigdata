import os
import pandas as pd

import matplotlib.pyplot as plt

def main():
    # Ruta de la carpeta que contiene los CSV
    folder_path = './resultados'
    
    # Obtener la lista de archivos CSV en la carpeta
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Leer cada archivo CSV y generar gráficos
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        
        # Leer el archivo CSV
        data = pd.read_csv(file_path)
        
        # Verificar que las columnas 'time' y 'n' existan
        if 'time' in data.columns and 'n' in data.columns:
            # Crear el gráfico
            plt.figure()
            print(data['n'], data['time'])
            plt.bar(data['n'], data['time'], color='blue', alpha=0.7)
            plt.title(f'Gráfico de {csv_file}')
            plt.xlabel('N')
            plt.ylabel('Time')
            plt.grid(True)
            
            # Guardar el gráfico como imagen
            output_path = os.path.join(folder_path, f'{csv_file}_grafico.png')
            plt.savefig(output_path)
            plt.close()
            print(f'Gráfico guardado en: {output_path}')
        else:
            print(f'El archivo {csv_file} no contiene las columnas necesarias.')

if __name__ == '__main__':
    main()