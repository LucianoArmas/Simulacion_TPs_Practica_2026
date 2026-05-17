"""
TP1.2 - Estudio Económico-Matemático de Apuestas en la Ruleta
---------------------------------
Integrantes:
- Tomás Lardizábal, legajo 47433
- Iñaki Díaz, legajo 48944
- Tomás Splivalo, legajo 51665
- Luciano Armas, legajo 47181
"""

import random, argparse, matplotlib.pyplot as plt, traceback, os

apuesta_inicial = 10


# --- FUNCIONES DE GRAFICACIÓN ---

def graficar_frecuencia_relativa_evolucion(ruta_carpeta, historial_fr, estrategia="", capital=""):
    # n_tiradas ahora se deduce del largo del historial
    total_tiradas = len(historial_fr)
    
    # Para que sea un gráfico de barras legible, seleccionamos 50 puntos representativos
    # Si graficamos todas las barras en 10.000 tiradas no se vería nada.
    paso = max(1, total_tiradas // 50) 
    indices = range(0, total_tiradas, paso)
    valores_fr = [historial_fr[i] for i in indices]
    
    plt.figure(figsize=(10, 5))
    
    # Grafico de barras: frsa (frecuencia relativa de obtener la apuesta favorable según n)
    plt.bar(indices, valores_fr, width=paso*0.8, color='skyblue', label='frsa (Simulada)', alpha=0.7)
    
    # Línea horizontal del valor esperado (Teórico)
    plt.axhline(y=1/37, color='red', linestyle='--', label='Valor Esperado (1/37)')
    
    plt.xlabel('n (Número de tiradas)')
    plt.ylabel('fr (Frecuencia Relativa)')
    plt.title('Frecuencia Relativa Acumulada (frsa) según n')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{ruta_carpeta}/frecuencia_relativa_evolucion_{estrategia}_{capital}.png", dpi=600, bbox_inches='tight')
    plt.show()

def graficar_frecuencias_multiples(ruta_carpeta, todas_las_fr, estrategia="", capital=""):
    plt.figure(figsize=(10, 5))
    for fr_corrida in todas_las_fr:
        plt.plot(fr_corrida, alpha=0.4)
    plt.axhline(y=1/37, color='black', linestyle='--', label='Valor Teórico')
    plt.xlabel('n (Número de tiradas)')
    plt.ylabel('fr (Frecuencia Relativa)')
    plt.title('Simultaneidad de Frecuencias Relativas')
    plt.savefig(f"{ruta_carpeta}/frecuencias_multiples_{estrategia}_{capital}.png", dpi=600, bbox_inches='tight')
    plt.show()



def graficar_flujo_caja(ruta_carpeta, historial_capital, capital_inicial, estrategia="", capital=""):
    plt.figure()
    # Gráfico azul: Flujo de caja inicial (línea constante)
    plt.axhline(y=capital_inicial, color='blue', label='FCI (Inicial)', linestyle='--')
    # Gráfico rojo: Flujo de caja (la evolución real)
    plt.plot(historial_capital, color='red', label='FC (Evolución)')
    
    plt.xlabel('Número de tiradas (n)')
    plt.ylabel('Cantidad de capital (cc)')
    plt.legend()
    plt.title('Evolución del Flujo de Caja')
    plt.savefig(f"{ruta_carpeta}/flujo_caja_{estrategia}_{capital}.png", dpi=600, bbox_inches='tight')
    plt.show()
    


def graficar_multiples_corridas(ruta_carpeta, lista_de_historiales, capital_inicial, estrategia="", capital=""):

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, historial in enumerate(lista_de_historiales):
        ax.plot(
            historial,
            alpha=0.6,
            label=f'Corrida {i + 1}'
        )

    # Línea del capital inicial
    ax.axhline(
        y=capital_inicial,
        color='blue',
        linestyle='--',
        label='FCI (Inicial)'
    )

    ax.set_title(f'Simultaneidad de {len(lista_de_historiales)} corridas')
    ax.set_xlabel('Tiradas')
    ax.set_ylabel('Capital')

    ax.legend()

    fig.savefig(f"{ruta_carpeta}/corridas_{len(lista_de_historiales)}_{estrategia}_{capital}.png", dpi=600, bbox_inches='tight')

    plt.show()
    plt.close(fig)



# --- ESTRATEGIAS DE APUESTA ---

def estrategia_martingala_invertida(beneficio, ultima_apuesta, is_win, capital):
    if is_win:
        capital += ultima_apuesta * beneficio
        proxima_apuesta = ultima_apuesta * 2
    else:
        capital -= ultima_apuesta 
        proxima_apuesta = apuesta_inicial
    return proxima_apuesta, capital


def estrategia_martingala(beneficio, ultima_apuesta, is_win, capital):
    if is_win:
        capital += ultima_apuesta * beneficio
        proxima_apuesta = apuesta_inicial
    else:
        capital -= ultima_apuesta
        # Limitar la apuesta para evitar que el programa explote y simular límite de mesa
        proxima_apuesta = min(ultima_apuesta * 2, 10**12)
    return proxima_apuesta, capital



def estrategia_dalembert(beneficio, ultima_apuesta, is_win, capital):
    unidad = apuesta_inicial
    if is_win:
        capital += ultima_apuesta * beneficio
        # Evitar apuestas menores que la unidad inicial
        proxima_apuesta = max(ultima_apuesta - unidad, unidad)
    else:
        capital -= ultima_apuesta
        proxima_apuesta = ultima_apuesta + unidad

    return proxima_apuesta, capital


def estrategia_fibonacci(beneficio, is_win, capital, apuesta_actual, indice, serie):
    if is_win:
        capital += apuesta_actual * beneficio
        # Al ganar, retrocedemos 2 posiciones
        nuevo_indice = max(0, indice - 2)
    else:
        capital -= apuesta_actual
        # Al perder, avanzamos 1 posición
        nuevo_indice = indice + 1
        
        # Si el índice supera el tamaño de la serie actual, la extendemos
        while nuevo_indice >= len(serie):
            serie.append(serie[-1] + serie[-2])

    # Limitar índice para evitar números gigantescos
    if nuevo_indice > 100: nuevo_indice = 100 
    proxima_apuesta = serie[nuevo_indice]
    return proxima_apuesta, capital, nuevo_indice, serie



# --- FUNCIONES PRINCIPALES ---

def ejecutar_simulacion(args):
  
  colores_negros = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}
  
  todas_las_corridas_capital = []
  frecuencias_finales = []
  todas_las_corridas_fr = []
  
  capital_inicial = 10000 if args.capital == 'f' else 0
  
  for i in range(args.corridas):
    capital = capital_inicial
    apuesta_actual = apuesta_inicial
    historial_capital = [capital]
    aciertos = 0
    
    historial_fr = []
    
    serie_fibo = [1, 1]
    idx_fibo = 0
    
    for t in range(1, args.tiradas + 1):
      
      if args.capital == "f" and capital < apuesta_actual:
        # Bancarrota
        # historial_capital.extend([0] * (args.tiradas - len(historial_capital) + 1))
        
        # # Para que la frecuencia no se rompa tras la bancarrota, mantenemos el último valor
        # ultima_fr = aciertos / (t-1) if t > 1 else 0
        # historial_fr.extend([ultima_fr] * (args.tiradas - len(historial_fr)))
        
        break
      
      resultado = random.randint(0, 36)
      
      if args.numero == -1:
        # APOSTAR A LOS NUMEROS "NEGROS":
        apuesta_favorable = resultado in colores_negros
        beneficio = 1
      else:
        # APOSTAR A UN UNICO NUMERO:
        apuesta_favorable = resultado == args.numero
        beneficio = 35

      if apuesta_favorable:
        aciertos += 1
        is_win = True
      else:
        is_win = False
      # Guardamos la frecuencia relativa en el momento T
      historial_fr.append(aciertos / t)
      
      if args.estrategia == 'd':
        apuesta_actual, capital = estrategia_dalembert(beneficio,apuesta_actual, is_win, capital)
        
      elif args.estrategia == 'f':
        apuesta_actual, capital, idx_fibo, serie_fibo = estrategia_fibonacci(beneficio,is_win, capital, apuesta_actual, idx_fibo, serie_fibo)
      
      elif args.estrategia == "m":
        apuesta_actual, capital = estrategia_martingala(beneficio,apuesta_actual, is_win, capital)
        
      elif args.estrategia == "o":
        apuesta_actual, capital = estrategia_martingala_invertida(beneficio,apuesta_actual, is_win, capital)
        
      
      historial_capital.append(capital)
      
    # Al terminar la corrida, guardamos los resultados globales
    todas_las_corridas_capital.append(historial_capital)
    frecuencias_finales.append(aciertos / args.tiradas)
    todas_las_corridas_fr.append(historial_fr)
    
  # return todas_las_corridas_capital, frecuencias_finales
  return todas_las_corridas_capital, todas_las_corridas_fr



def get_args():
    parser = argparse.ArgumentParser(description='TP Simulación - Ruleta')
    parser.add_argument('-c', '--tiradas', type=int, required=True, help='Cantidad de tiradas por corrida')
    parser.add_argument('-n', '--numero', type=int, required=True, help='Número elegido (0-36)')
    parser.add_argument('-e', '--corridas', type=int, required=True, help='Cantidad de corridas a simular')
    parser.add_argument('-s', '--estrategia', type=str, help='Estrategia a utilizar')
    parser.add_argument('-a', '--capital', type=str, help='Tipo de capital')
    return parser.parse_args()




# --- GESTOR DE CARPETAS PARA GUARDAR GRÁFICOS ---
CAPITAL_MAP = {
    'f': 'capital_finito',
    'i': 'capital_infinito'
}

ESTRATEGIA_MAP = {
    'd': 'dalembert',
    'f': 'fibonacci',
    'm': 'martingala',
    'o': 'martingala_invertida'
}

def get_carpeta(estrategia, capital, numero):
    carpeta_estrategia = ESTRATEGIA_MAP.get(estrategia, 'desconocida')
    carpeta_capital = CAPITAL_MAP.get(capital, 'desconocido')
    
    if numero == -1:
        carpeta_estrategia += "_numeros_negros"
    else:
        carpeta_estrategia += f"_numero_unico"
        
    ruta_carpeta = os.path.join("graficos", carpeta_capital, carpeta_estrategia)
    os.makedirs(ruta_carpeta, exist_ok=True)
    
    return ruta_carpeta


def main():
  try:
    args = get_args()

    # 1. Correr la simulación y obtener datos
    datos_capital, datos_frecuencias = ejecutar_simulacion(args)

    # 2. Definir capital inicial para los gráficos
    cap_init = 10000 if args.capital == 'f' else 0

    # 3. Preparar carpeta para guardar gráficos
    ruta_carpeta = get_carpeta(args.estrategia, args.capital, args.numero)

    # 4. Graficar resultados
    # Gráfico 1: Frecuencia Relativa 
    graficar_frecuencia_relativa_evolucion(ruta_carpeta, datos_frecuencias[0], args.estrategia, args.capital)

    # Gráfico 2: Flujo de Caja (solo de la primera corrida para que sea legible)
    graficar_flujo_caja(ruta_carpeta, datos_capital[0], cap_init, args.estrategia, args.capital)

    # Gráfico 3: Multiples Corridas (todas juntas)
    graficar_multiples_corridas(ruta_carpeta, datos_capital, cap_init, args.estrategia, args.capital)

    # Gráfico 4: Grafico de Frecuencias Simultáneas
    graficar_frecuencias_multiples(ruta_carpeta, datos_frecuencias, args.estrategia, args.capital)

  except Exception as e:
    print(f"Error: {e}")
    print("Traceback:")
    traceback.print_exc()
    


if __name__ == "__main__":
    main()

# parametro de ejemplo para ejecutar el programa:
# python ruleta.py -c 10000 -n 11 -e 5 -s 'd' -a 'f'

# c: cantidad giros de ruleta, n: número elegido, e: cantidad de corridas a graficar, s: estrategia a utilizar, a: tipo de capital

# s: 'd' para D'Alembert, 'f' para Fibonacci, 'm' para Martingala, 'o' para Martingala Invertida
# a: 'f' para capital finito, 'i' para capital infinito

