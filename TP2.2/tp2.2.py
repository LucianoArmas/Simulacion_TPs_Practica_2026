"""
TP2.2 - Generadores de Numeros Pseudoaleatorios de Distintas Probabilidades
---------------------------------
Integrantes:
- Tomás Lardizábal, legajo 47433
- Tomás Splivalo, legajo 51665
- Luciano Armas, legajo 47181
"""

import random
import math
import matplotlib.pyplot as plt

def u():
    """Devuelve un número pseudoaleatorio uniforme en (0,1)."""
    return random.random()

# ===========================================================================
#  DISTRIBUCIONES CONTINUAS
# ===========================================================================

def uniforme_inversa(a, b):
    """Transformación inversa. F(x) = (x-a)/(b-a) => x = a + (b-a)*r"""
    return a + (b - a) * u()

def uniforme_rechazo(a, b):
    """Método de rechazo para uniforme."""
    c = b - a
    f = 1.0 / (b - a)
    while True:
        r1 = u()
        r2 = u()
        x = a + (b - a) * r1
        if r2 <= c * f:
            return x

def exponencial_inversa(ex):
    """Transformación inversa para Exponencial. media = ex = 1/alpha."""
    return -ex * math.log(u())

def exponencial_von_neumann(ex=1.0):
    """Método de rechazo de Von Neumann, sin usar logaritmo."""
    k = 0
    while True:
        u0 = u()
        actual = u0
        largo = 1
        while True:
            siguiente = u()
            if siguiente >= actual:
                break
            actual = siguiente
            largo += 1
        if largo % 2 == 1:
            return ex * (k + u0)
        k += 1

def gamma_erlang(alpha, k):
    """Distribución Erlang: suma de k exponenciales."""
    prod = 1.0
    for _ in range(k):
        prod *= u()
    return -math.log(prod) / alpha

def gamma_parametros(ex, vx):
    """Deriva (alpha, k) a partir de la media y la varianza."""
    alpha = ex / vx
    k = round(ex * ex / vx)
    return alpha, k

def normal_limite_central(mu, sigma):
    """Método del límite central con K=12."""
    s = 0.0
    for _ in range(12):
        s += u()
    return sigma * (s - 6.0) + mu

def normal_box_muller(mu, sigma):
    """Procedimiento directo de Box-Muller."""
    r1 = u()
    r2 = u()
    z = math.sqrt(-2.0 * math.log(r1)) * math.cos(2.0 * math.pi * r2)
    return mu + sigma * z

# ===========================================================================
#  DISTRIBUCIONES DISCRETAS
# ===========================================================================

def pascal(k, p):
    """Distribución de Pascal (Suma de k geométricas)."""
    q = 1.0 - p
    lnq = math.log(q)
    x = 0
    for _ in range(k):
        x += int(math.log(u()) / lnq)
    return x

def binomial(n, p):
    """Reproducción de n ensayos de Bernoulli."""
    x = 0
    for _ in range(n):
        if u() <= p:
            x += 1
    return x

def hipergeometrica(N, n, p):
    """Muestreo SIN reemplazo."""
    TN = float(N)
    P = p
    x = 0
    for _ in range(n):
        if u() <= P:
            S = 1.0
            x += 1
        else:
            S = 0.0
        P = (TN * P - S) / (TN - 1.0)
        TN -= 1.0
    return x

def poisson(lam):
    """Producto de uniformes comparado con e^(-lambda)."""
    B = math.exp(-lam)
    tr = 1.0
    x = 0
    while True:
        tr *= u()
        if tr < B:
            return x
        x += 1

def empirica(valores, probs):
    """Transformación inversa sobre la función acumulada."""
    r = u()
    acum = 0.0
    for v, pr in zip(valores, probs):
        acum += pr
        if r <= acum:
            return v
    return valores[-1]

# ===========================================================================
#  HERRAMIENTAS DE ESTADÍSTICA Y GRÁFICOS
# ===========================================================================
def estadisticas(muestra):
    n = len(muestra)
    media = sum(muestra) / n
    var = sum((x - media) ** 2 for x in muestra) / n
    return media, var

def comparar(ex_teo, vx_teo, media, var):
    err_m = abs(media - ex_teo) / (abs(ex_teo) + 1e-12)
    err_v = abs(var - vx_teo) / (abs(vx_teo) + 1e-12)
    ok = "OK" if err_m < 0.03 and err_v < 0.06 else "revisar"
    print(f"    {'':12}{'teórico':>12}{'empírico':>12}{'error rel.':>12}")
    print(f"    {'media (EX)':12}{ex_teo:>12.4f}{media:>12.4f}{err_m:>11.2%}")
    print(f"    {'varianza(VX)':12}{vx_teo:>12.4f}{var:>12.4f}{err_v:>11.2%}")
    print(f"    -> validación: {ok}")
    return ok == "OK"

def graficar_y_guardar(muestra, titulo, nombre_archivo, es_continua=True, x_teorico=None, y_teorico=None):
    """Genera el histograma empírico, superpone la curva teórica y guarda en PNG."""
    plt.figure(figsize=(10, 5))
    
    if es_continua:
        # density=True hace que el área total sea 1, permitiendo comparar con la FDP teórica
        plt.hist(muestra, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='white', label='Empírico (Muestra)')
        if x_teorico and y_teorico:
            plt.plot(x_teorico, y_teorico, color='darkorange', linewidth=2, label='Teórico (FDP)')
    else:
        # Para discretas calculamos la proporción real de cada entero observado
        valores_unicos = sorted(list(set(muestra)))
        frec_relativas = [muestra.count(v) / len(muestra) for v in valores_unicos]
        plt.bar(valores_unicos, frec_relativas, alpha=0.6, color='seagreen', edgecolor='white', width=0.4, label='Empírico (Muestra)')
        if x_teorico and y_teorico:
            plt.stem(x_teorico, y_teorico, linefmt='r-', markerfmt='ro', basefmt=' ', label='Teórico (PMF)')
            
    plt.title(titulo, fontsize=12, fontweight='bold')
    plt.xlabel('Valor de la Variable (X)')
    plt.ylabel('Frecuencia Relativa / Densidad')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Guardar el archivo en alta definición
    plt.savefig(f"./TP2.2/figuras/{nombre_archivo}", dpi=300, bbox_inches='tight')
    print(f"    -> Gráfico guardado como '{nombre_archivo}'")
    plt.close() # Importante para liberar memoria RAM

def encabezado(titulo):
    print("\n" + "=" * 78)
    print("  " + titulo)
    print("=" * 78)

# Funciones de probabilidad teóricas para las curvas
def pmf_binomial(n, p):
    q = 1 - p
    return {x: math.comb(n, x) * p**x * q**(n - x) for x in range(n + 1)}

# ===========================================================================
#  PROGRAMA PRINCIPAL
# ===========================================================================
def main():
    random.seed(2026)          # reproducibilidad
    N = 100000                 # tamaño de muestra
    resultados = []

    # 1. DISTRIBUCIÓN UNIFORME
    encabezado("UNIFORME (a=2, b=8) [inversa]")
    a, b = 2, 8
    ex, vx = (a + b) / 2, (b - a) ** 2 / 12
    m_inv = [uniforme_inversa(a, b) for _ in range(N)]
    resultados.append(comparar(ex, vx, *estadisticas(m_inv)))
    
    # Curva teórica Uniforme
    x_t = [a, a, b, b]
    y_t = [0, 1/(b-a), 1/(b-a), 0]
    graficar_y_guardar(m_inv, "Distribución Uniforme (a=2, b=8)", "dist_uniforme.png", True, x_t, y_t)

    # 2. DISTRIBUCIÓN EXPONENCIAL
    encabezado("EXPONENCIAL (EX=3) [inversa]")
    ex_val = 3.0
    vx_val = ex_val ** 2
    m_exp = [exponencial_inversa(ex_val) for _ in range(N)]
    resultados.append(comparar(ex_val, vx_val, *estadisticas(m_exp)))
    
    # Curva teórica Exponencial
    max_exp = max(m_exp)
    x_t = [max_exp * i / 200 for i in range(201)]
    lam = 1.0 / ex_val
    y_t = [lam * math.exp(-lam * x) for x in x_t]
    graficar_y_guardar(m_exp, "Distribución Exponencial (Media=3)", "dist_exponencial.png", True, x_t, y_t)

    # 3. DISTRIBUCIÓN GAMMA (ERLANG)
    encabezado("GAMMA / ERLANG (EX=2, VX=1)")
    ex_g, vx_g = 2.0, 1.0
    alpha, k = gamma_parametros(ex_g, vx_g)
    m_gamma = [gamma_erlang(alpha, k) for _ in range(N)]
    resultados.append(comparar(ex_g, vx_g, *estadisticas(m_gamma)))
    
    # Curva teórica Gamma
    max_g = max(m_gamma)
    x_t = [max_g * i / 200 for i in range(201)]
    y_t = []
    for x in x_t:
        f = ((alpha**k) / math.factorial(k-1)) * (x**(k-1)) * math.exp(-alpha * x) if x > 0 else 0
        y_t.append(f)
    graficar_y_guardar(m_gamma, f"Distribución Gamma Erlang (alpha={alpha}, k={k})", "dist_gamma.png", True, x_t, y_t)

    # 4. DISTRIBUCIÓN BINOMIAL
    encabezado("BINOMIAL (n=20, p=0.3)")
    n_b, p_b = 20, 0.3
    q_b = 1 - p_b
    ex_b, vx_b = n_b * p_b, n_b * p_b * q_b
    m_bin = [binomial(n_b, p_b) for _ in range(N)]
    resultados.append(comparar(ex_b, vx_b, *estadisticas(m_bin)))
    
    # Curva teórica Binomial
    pmf = pmf_binomial(n_b, p_b)
    x_t = list(pmf.keys())
    y_t = list(pmf.values())
    graficar_y_guardar(m_bin, "Distribución Binomial (n=20, p=0.3)", "dist_binomial.png", False, x_t, y_t)

    print("\n" + "=" * 78)
    print(f"  Proceso terminado. Gráficos exportados con éxito.")
    print("=" * 78)

if __name__ == "__main__":
    main()








