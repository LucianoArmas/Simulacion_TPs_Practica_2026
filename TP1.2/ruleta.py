import random, argparse, matplotlib.pyplot as plt

apuesta_inicial = 1

def graficar_frecuencia_relativa(fr_obtenida):
    fr_esperada = 1/37 
    plt.figure(figsize=(8, 5))
    plt.bar(['Simulada (frsa)', 'Teórica'], [fr_obtenida, fr_esperada], color=['blue', 'gray'])
    plt.axhline(y=fr_esperada, color='red', linestyle='--') # Línea de referencia
    plt.ylabel('Frecuencia Relativa')
    plt.title('Frecuencia Relativa de obtener la apuesta favorable')
    plt.show()



def graficar_flujo_caja(historial_capital, capital_inicial):
    plt.figure()
    # Gráfico azul: Flujo de caja inicial (línea constante)
    plt.axhline(y=capital_inicial, color='blue', label='FCI (Inicial)', linestyle='--')
    # Gráfico rojo: Flujo de caja (la evolución real)
    plt.plot(historial_capital, color='red', label='FC (Evolución)')
    
    plt.xlabel('Número de tiradas (n)')
    plt.ylabel('Cantidad de capital (cc)')
    plt.legend()
    plt.title('Evolución del Flujo de Caja')
    plt.show()

def graficar_multiples_corridas(lista_de_historiales, capital_inicial):
    plt.figure()
    for historial in lista_de_historiales:
        plt.plot(historial, alpha=0.6) # alpha para que se vean todas si se superponen
    
    # Gráfico azul: Flujo de caja inicial (línea constante)
    plt.axhline(y=capital_inicial, color='blue', label='FCI (Inicial)', linestyle='--')
    
    plt.title(f'Simultaneidad de {len(lista_de_historiales)} corridas')
    plt.xlabel('Tiradas')
    plt.ylabel('Capital')
    plt.show()



def get_args():
    parser = argparse.ArgumentParser(description='TP Simulación - Ruleta')
    parser.add_argument('-c', '--tiradas', type=int, required=True, help='Cantidad de tiradas por corrida')
    parser.add_argument('-n', '--numero', type=int, required=True, help='Número elegido (0-36)')
    parser.add_argument('-e', '--corridas', type=int, required=True, help='Cantidad de corridas a simular')
    parser.add_argument('-s', '--estrategia', type=str, help='Estrategia a utilizar')
    parser.add_argument('-a', '--capital', type=str, help='Tipo de capital')
    return parser.parse_args()


def estrategia_martingala_invertida(ultima_apuesta, is_win, capital):
    if is_win:
        capital += ultima_apuesta * 35
        proxima_apuesta = ultima_apuesta * 2
    else:
        capital -= ultima_apuesta 
        proxima_apuesta = apuesta_inicial
    return proxima_apuesta, capital


def estrategia_martingala(ultima_apuesta, is_win, capital):
    if is_win:
        capital += ultima_apuesta * 35
        proxima_apuesta = apuesta_inicial
    else:
        capital -= ultima_apuesta
        proxima_apuesta = ultima_apuesta * 2
    return proxima_apuesta, capital



def estrategia_dalembert(ultima_apuesta, is_win, capital):
    unidad = apuesta_inicial
    if is_win:
        capital += ultima_apuesta * 35
        # Evitar apuestas menores que la unidad inicial
        proxima_apuesta = max(ultima_apuesta - unidad, unidad)
    else:
        capital -= ultima_apuesta
        proxima_apuesta = ultima_apuesta + unidad

    return proxima_apuesta, capital


def estrategia_fibonacci(is_win, capital, apuesta_actual, indice, serie):
    if is_win:
        capital += apuesta_actual * 35
        # Al ganar, retrocedemos 2 posiciones
        nuevo_indice = max(0, indice - 2)
    else:
        capital -= apuesta_actual
        # Al perder, avanzamos 1 posición
        nuevo_indice = indice + 1
        
        # Si el índice supera el tamaño de la serie actual, la extendemos
        while nuevo_indice >= len(serie):
            serie.append(serie[-1] + serie[-2])

    proxima_apuesta = serie[nuevo_indice]
    return proxima_apuesta, capital, nuevo_indice, serie



def ejecutar_simulacion(args):
  todas_las_corridas_capital = []
  frecuencias_finales = []
  
  capital_inicial = 1000 if args.capital == 'f' else 10000000000
  
  for i in range(args.corridas):
    capital = capital_inicial
    apuesta_actual = apuesta_inicial
    historial_capital = [capital]
    aciertos = 0
    
    serie_fibo = [1, 1]
    idx_fibo = 0
    
    for t in range(1, args.tiradas + 1):
      
      if args.capital == "f" and capital < apuesta_actual:
        # Bancarrota
        historial_capital.extend([0] * (args.tiradas - len(historial_capital) + 1))
        break
      
      resultado = random.randint(0, 36)
      
      if resultado == args.numero:
        aciertos += 1
        is_win = True
      else:
        is_win = False
      
      if args.estrategia == 'd':
        apuesta_actual, capital = estrategia_dalembert(apuesta_actual, is_win, capital)
        
      elif args.estrategia == 'f':
        apuesta_actual, capital, idx_fibo, serie_fibo = estrategia_fibonacci(is_win, capital, apuesta_actual, idx_fibo, serie_fibo)
      
      elif args.estrategia == "m":
        apuesta_actual, capital = estrategia_martingala(apuesta_actual, is_win, capital)
        
      elif args.estrategia == "o":
        apuesta_actual, capital = estrategia_martingala_invertida(apuesta_actual, is_win, capital)
        
      
      historial_capital.append(capital)
      
    # Al terminar la corrida, guardamos los resultados globales
    todas_las_corridas_capital.append(historial_capital)
    frecuencias_finales.append(aciertos / args.tiradas)
    
  return todas_las_corridas_capital, frecuencias_finales




def main():
  args = get_args()
  
  # 1. Correr la simulación y obtener datos
  datos_capital, datos_frecuencias = ejecutar_simulacion(args)

  # 2. Definir capital inicial para los gráficos
  cap_init = 1000 if args.capital == 'f' else 10000000000

  # 3. Graficar resultados
  # Gráfico 1: Frecuencia Relativa (puedes usar el promedio de todas las corridas)
  fr_promedio = sum(datos_frecuencias) / len(datos_frecuencias)
  graficar_frecuencia_relativa(fr_promedio)
  
  # Gráfico 2: Flujo de Caja (solo de la primera corrida para que sea legible)
  graficar_flujo_caja(datos_capital[0], cap_init)
  
  # Gráfico 3: Multiples Corridas (todas juntas)
  graficar_multiples_corridas(datos_capital, cap_init)
  


if __name__ == "__main__":
    main()

# parametro de ejemplo para ejecutar el programa:
# python ruleta.py -c 10000 -n 11 -e 1 -s 'd' -a 'f'
# c: cantidad giros de ruleta, n: número elegido, e: cantidad de corridas a graficar, s: estrategia a utilizar, a: tipo de capital
