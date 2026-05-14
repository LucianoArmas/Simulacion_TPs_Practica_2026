"""
TP1.1 - Simulación de la Ruleta
---------------------------------
Integrantes:
- Tomás Lardizábal, legajo 47433
- Iñaki Díaz, legajo 48944
- Tomás Splivalo, legajo 51665
- Luciano Armas, legajo 47181
"""



import argparse
import random
import matplotlib.pyplot as plt
import numpy as np

def simular_ruleta(tiradas, numero_elegido):
    datos = []
    frec_favorables = 0
    fr_n, vp_n, vv_n, vd_n = [], [], [], []

    for i in range(1, tiradas + 1):
        resultado = random.randint(0, 36)
        datos.append(resultado)
        
        # 1. Frecuencia Relativa
        if resultado == numero_elegido:
            frec_favorables += 1
        fr_n.append(frec_favorables / i)
        
        # 2. Media, Varianza y Desvío
        # Usamos numpy sobre la lista actual
        arr_actual = np.array(datos)
        media = np.mean(arr_actual)
        varianza = np.var(arr_actual)
        
        vp_n.append(media)
        vv_n.append(varianza)
        vd_n.append(np.sqrt(varianza))
        
    return fr_n, vp_n, vv_n, vd_n, datos

def graficar_metrica(ax, datos_corridas, valor_esperado, titulo, ylabel, colores, mostrar_promedio=False):
    """Función auxiliar para no repetir código de gráficas"""
    # Convertimos a matriz de numpy para promediar fácilmente: [corridas, tiradas]
    matriz = np.array(datos_corridas)
    
    if mostrar_promedio:
        # Promediamos verticalmente (promedio de todas las corridas en cada instante n)
        promedio_final = np.mean(matriz, axis=0)
        ax.plot(promedio_final, color='darkred', linewidth=2, label='Promedio de Corridas')
    else:
        # Graficamos cada corrida individual
        for i in range(matriz.shape[0]):
            # ax.plot(matriz[i], color=color_lineas, alpha=0.4, linewidth=0.8)
            ax.plot(matriz[i], color=colores[i], alpha=0.7, linewidth=1, label=f'Corrida {i+1}')

    ax.axhline(y=valor_esperado, color='blue', linestyle='--', label=f'Esperado: {valor_esperado:.4f}')
    ax.set_title(titulo)
    ax.set_xlabel('n (tiradas)')
    ax.set_ylabel(ylabel)
    
    # ax.legend(fontsize='small')
    # Solo mostramos leyenda si hay pocas corridas o es el promedio, para no tapar el gráfico
    if matriz.shape[0] <= 10 or mostrar_promedio:
        ax.legend(fontsize='x-small', loc='best')
    
    ax.grid(True, alpha=0.3)

def graficar_histograma(todos_los_resultados):
    plt.figure(figsize=(10, 5))
    # Aplanamos la lista de listas para tener todos los números que salieron
    flat_data = [item for sublist in todos_los_resultados for item in sublist]
    
    plt.hist(flat_data, bins=37, range=(0, 37), density=True, color='seagreen', edgecolor='white', alpha=0.7)
    plt.axhline(y=1/37, color='red', linestyle='--', label='Frecuencia relativa esperada (1/37)')
    plt.title('Histograma de Frecuencias (Validación de Distribución Uniforme)')
    plt.xlabel('Número')
    plt.ylabel('Frecuencia Relativa')
    plt.legend()
    plt.show()
    
def graficar_histogramas_multiples(all_raw, colores):
    plt.figure(figsize=(12, 6))
    # Cantidad de números en la ruleta
    n_bins = 37
    
    # Graficamos el histograma de cada corrida
    for i, data in enumerate(all_raw):
        plt.hist(data, bins=n_bins, range=(0, 37), density=True, 
                color=colores[i], alpha=0.3, edgecolor='white', label=f'Corrida {i+1}')
    
    plt.axhline(y=1/37, color='black', linestyle='--', linewidth=2, label='Teórico (1/37)')
    plt.title('Histogramas Individuales por Corrida (Frecuencias de cada número)')
    plt.xlabel('Número')
    plt.ylabel('Frecuencia Relativa')
    if len(all_raw) <= 15:
        plt.legend(fontsize='x-small', ncol=2)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='TP Simulación - Ruleta')
    parser.add_argument('-c', '--tiradas', type=int, required=True)
    parser.add_argument('-n', '--numero', type=int, required=True)
    parser.add_argument('-e', '--corridas', type=int, required=True)
    args = parser.parse_args()

    # Listas para almacenar los resultados de cada métrica por corrida
    all_fr, all_vp, all_vv, all_vd, all_raw = [], [], [], [], []

    for _ in range(args.corridas):
        fr, vp, vv, vd, raw = simular_ruleta(args.tiradas, args.numero)
        all_fr.append(fr)
        all_vp.append(vp)
        all_vv.append(vv)
        all_vd.append(vd)
        all_raw.append(raw)

    # Generamos una paleta de colores dinámicos (usamos 'tab10' o 'viridis' según la cantidad)
    cmap = plt.get_cmap('tab10') if args.corridas <= 10 else plt.get_cmap('viridis')
    colores = [cmap(i / args.corridas) for i in range(args.corridas)]

    # Valores Teóricos
    teoricos = [1/37, 18, np.var(np.arange(37)), np.std(np.arange(37))]
    nombres = [("Frecuencia Relativa", "fr (frecuencia relativa)"), ("Esperanza Matemática", "vp (valor promedio)"), ("Varianza", "vv (valor de la varianza)"), ("Desvío Estándar", "vd (valor del desvío)")]
    
    # --- FIGURA 1: CORRIDAS INDIVIDUALES (4 Gráficas) ---
    fig1, axs1 = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle('Resultados por Corrida Individual', fontsize=16)
    data_list = [all_fr, all_vp, all_vv, all_vd]
    
    for i in range(4):
        graficar_metrica(axs1[i//2, i%2], data_list[i], teoricos[i], nombres[i][0], nombres[i][1], colores)

    # --- FIGURA 2: PROMEDIO DE CORRIDAS (4 Gráficas) ---
    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle('Promedio de Todas las Corridas', fontsize=16)
    
    for i in range(4):
        graficar_metrica(axs2[i//2, i%2], data_list[i], teoricos[i], nombres[i][0], nombres[i][1], colores, mostrar_promedio=True)

    plt.show()

    # --- FIGURA 3: HISTOGRAMA ---
    graficar_histogramas_multiples(all_raw, colores)
    
    graficar_histograma(all_raw)

if __name__ == "__main__":
    main()
    
# parametro de ejemplo para ejecutar el programa:
# python ruleta.py -c 10000 -n 11 -e 10
# c: cantidad giros de ruleta, n: número elegido, e: cantidad de corridas a graficar
