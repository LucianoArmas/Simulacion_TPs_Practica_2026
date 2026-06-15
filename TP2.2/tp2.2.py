"""
TP2.2 - Generadores de Numeros Pseudoaleatorios de Distintas Probabilidades
---------------------------------
Integrantes:
- Tomás Lardizábal, legajo 47433
- Tomás Splivalo, legajo 51665
- Luciano Armas, legajo 47181
"""


import math
import random
import matplotlib.pyplot as plt

# ===========================================================================
#  FUENTE UNIFORME BASE  U(0,1)
# ===========================================================================
def u():
    """
    Devuelve un número pseudoaleatorio uniforme en (0,1).
    """
    return random.random()


# ===========================================================================
#  DISTRIBUCIONES CONTINUAS
# ===========================================================================

# ---------------------------- UNIFORME (a, b) ------------------------------
def uniforme_inversa(a, b):
    """Transformación inversa.  F(x) = (x-a)/(b-a)  =>  x = a + (b-a)*r."""
    return a + (b - a) * u()


def uniforme_rechazo(a, b):
    """Método de rechazo.

    f(x) = 1/(b-a) constante. Se escala con c = (b-a) para que c*f(x) = 1.
    x = a + (b-a)*r1 ;  se acepta si  r2 <= c*f(x).  Como c*f(x)=1, siempre
    acepta: el rechazo no aporta eficiencia aquí, pero ilustra el mecanismo.
    """
    c = b - a
    f = 1.0 / (b - a)
    while True:
        r1 = u()
        r2 = u()
        x = a + (b - a) * r1
        if r2 <= c * f:
            return x


# ---------------------------- EXPONENCIAL (EX) -----------------------------
def exponencial_inversa(ex):
    """Transformación inversa.  r = e^(-alpha*x)  =>  x = -EX*ln(r).
    ex = EX = media = 1/alpha."""
    return -ex * math.log(u())


def exponencial_von_neumann(ex=1.0):
    """Método de rechazo de Von Neumann, sin usar logaritmo.

    Genera corridas decrecientes de uniformes y acepta según su paridad.
    Produce Exp(media=1) y luego se escala por EX.
    """
    k = 0
    while True:
        u0 = u()
        actual = u0
        largo = 1
        while True:
            siguiente = u()
            if siguiente >= actual:      # primer ascenso: termina la corrida
                break
            actual = siguiente
            largo += 1
        if largo % 2 == 1:               # corrida de largo impar -> aceptar
            return ex * (k + u0)
        k += 1                           # par -> sumar 1 a la parte entera


# ------------------------- GAMMA / ERLANG (alpha, k) -----------------------
def gamma_erlang(alpha, k):
    """Distribución gamma con k entero (Erlang): suma de k exponenciales.

    x = -(1/alpha) * ln( PROD r_i ).
    EX = k/alpha ,  VX = k/alpha^2.
    """
    prod = 1.0
    for _ in range(k):
        prod *= u()
    return -math.log(prod) / alpha


def gamma_parametros(ex, vx):
    """Deriva (alpha, k) a partir de la media y la varianza."""
    alpha = ex / vx
    k = round(ex * ex / vx)
    return alpha, k


# ------------------------------ NORMAL (mu, sigma) -------------------------
def normal_limite_central(mu, sigma):
    """Método del límite central con K=12.

    x = sigma * (SUM_{i=1}^{12} r_i - 6) + mu.
    Con K=12 se evita la multiplicación por sqrt(12/K) y queda truncada a +/-6 sigma.
    """
    s = 0.0
    for _ in range(12):
        s += u()
    return sigma * (s - 6.0) + mu


def normal_box_muller(mu, sigma):
    """Procedimiento directo de Box-Muller, resultado exacto."""
    r1 = u()
    r2 = u()
    z = math.sqrt(-2.0 * math.log(r1)) * math.cos(2.0 * math.pi * r2)
    return mu + sigma * z


# ===========================================================================
#  DISTRIBUCIONES DISCRETAS
# ===========================================================================

# ------------------------------ PASCAL (k, p) ------------------------------
def pascal(k, p):
    """Distribución de Pascal (binomial negativa con k entero).

    Es la SUMA de k variables geométricas, cada una generada por inversa:  x_j = floor( ln(r_j) / ln(q) ).  Cuenta el número de fracasos
    antes del k-ésimo éxito.   EX = k*q/p ,  VX = k*q/p^2   (q = 1 - p).

    Nota: la forma compacta de Naylor escribe x = floor(ln(PROD r_i)/ln q),
    pero floor(suma) != suma(floor); esa versión truncada NO reproduce la
    binomial negativa (sesga la media). Por eso aquí se suman k geométricas
    truncadas individualmente, que es lo correcto.
    """
    q = 1.0 - p
    lnq = math.log(q)
    x = 0
    for _ in range(k):
        x += int(math.log(u()) / lnq)     # geométrica
    return x


# ----------------------------- BINOMIAL (n, p) -----------------------------
def binomial(n, p):
    """Reproducción de n ensayos de Bernoulli.

    Cuenta cuántos de los n uniformes cumplen r_i <= p (éxitos).
    EX = n*p ,  VX = n*p*q.
    """
    x = 0
    for _ in range(n):
        if u() <= p:
            x += 1
    return x


# ----------------------- HIPERGEOMÉTRICA (N, n, p) -------------------------
def hipergeometrica(N, n, p):
    """Muestreo SIN reemplazo.

    N = tamaño de la población, n = tamaño de la muestra,
    p = proporción inicial de la clase I (N*p elementos clase I).
    Tras cada extracción se actualizan p y N de forma dependiente.
    EX = n*p ,  VX = n*p*q*(N-n)/(N-1).
    """
    TN = float(N)
    P = p
    x = 0
    for _ in range(n):
        if u() <= P:          # se extrajo un elemento de la clase I
            S = 1.0
            x += 1
        else:
            S = 0.0
        P = (TN * P - S) / (TN - 1.0)   # nueva proporción de clase I
        TN -= 1.0                        # queda un elemento menos
    return x


# ------------------------------ POISSON (lambda) ---------------------------
def poisson(lam):
    """Producto de uniformes comparado con e^(-lambda).

    Se multiplican uniformes mientras el producto sea >= e^(-lambda);
    x es la cantidad de multiplicaciones realizadas menos uno.
    EX = VX = lambda.  (Para lambda > ~10 conviene la aprox. normal.)
    """
    B = math.exp(-lam)
    tr = 1.0
    x = 0
    while True:
        tr *= u()
        if tr < B:
            return x
        x += 1


# --------------------------- EMPÍRICA DISCRETA -----------------------------
def empirica(valores, probs):
    """Transformación inversa sobre la función acumulada.

    Se busca el i tal que  P_1+..+P_{i-1} < r <= P_1+..+P_i  y se devuelve b_i.
    """
    r = u()
    acum = 0.0
    for v, pr in zip(valores, probs):
        acum += pr
        if r <= acum:
            return v
    return valores[-1]   # protección por redondeo


# ===========================================================================
#  HERRAMIENTAS DE TESTEO
# ===========================================================================
def estadisticas(muestra):
    """Media y varianza muestral (varianza poblacional, /N)."""
    n = len(muestra)
    media = sum(muestra) / n
    var = sum((x - media) ** 2 for x in muestra) / n
    return media, var


def _barra(frac, ancho=50):
    return "#" * int(round(frac * ancho))


def histograma_continua(muestra, bins=20, ancho=45):
    """Histograma ASCII para variables continuas."""
    lo, hi = min(muestra), max(muestra)
    if hi == lo:
        hi = lo + 1e-9
    paso = (hi - lo) / bins
    cuentas = [0] * bins
    for x in muestra:
        i = int((x - lo) / paso)
        if i == bins:
            i -= 1
        cuentas[i] += 1
    pico = max(cuentas) or 1
    print("    intervalo                 frec")
    for i, c in enumerate(cuentas):
        a = lo + i * paso
        b = a + paso
        print(f"    [{a:8.3f},{b:8.3f}) {_barra(c / pico, ancho):<{ancho}} {c}")


def histograma_discreta(muestra, probs_teoricas, ancho=40):
    """Compara frecuencia empírica vs probabilidad teórica para variables discretas.

    probs_teoricas: dict {valor: probabilidad}.
    """
    n = len(muestra)
    valores = sorted(probs_teoricas.keys())
    print(f"    {'valor':>6} {'p.teor':>8} {'p.emp':>8}   histograma (emp)")
    chi = 0.0
    for v in valores:
        obs = sum(1 for x in muestra if x == v)
        p_emp = obs / n
        p_teo = probs_teoricas[v]
        esp = p_teo * n
        if esp > 0:
            chi += (obs - esp) ** 2 / esp
        print(f"    {v:>6} {p_teo:>8.4f} {p_emp:>8.4f}   {_barra(p_emp / max(probs_teoricas.values()), ancho)}")
    df = len(valores) - 1
    print(f"    Chi-cuadrado = {chi:.3f}  (gl = {df})")


def encabezado(titulo):
    print("\n" + "=" * 78)
    print("  " + titulo)
    print("=" * 78)


def comparar(ex_teo, vx_teo, media, var):
    """Imprime tabla teórico vs empírico y marca OK si el error es chico."""
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
        
        ancho = 0.4 # Ancho de cada barra
        
        if x_teorico and y_teorico:
            # Desplazamos las barras empíricas a la izquierda y las teóricas a la derecha
            x_emp = [v - ancho/2 for v in valores_unicos]
            x_teo = [v + ancho/2 for v in x_teorico]
            
            plt.bar(x_emp, frec_relativas, alpha=0.7, color='seagreen', edgecolor='white', width=ancho, label='Empírico (Muestra)')
            plt.bar(x_teo, y_teorico, alpha=0.7, color='darkorange', edgecolor='white', width=ancho, label='Teórico (PMF)')
        else:
            # Comportamiento por defecto si no hay teórico
            plt.bar(valores_unicos, frec_relativas, alpha=0.7, color='seagreen', edgecolor='white', width=ancho, label='Empírico (Muestra)')
            
    plt.title(titulo, fontsize=12, fontweight='bold')
    plt.xlabel('Valor de la Variable (X)')
    plt.ylabel('Frecuencia Relativa / Probabilidad')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Guardar el archivo en alta definición
    plt.savefig(f"./TP2.2/figuras/{nombre_archivo}", dpi=300, bbox_inches='tight')
    print(f"    -> Gráfico guardado como '{nombre_archivo}'")
    plt.close()



# ----------------- distribuciones de probabilidad teóricas -----------------
def pmf_binomial(n, p):
    q = 1 - p
    return {x: math.comb(n, x) * p**x * q**(n - x) for x in range(n + 1)}


def pmf_poisson(lam, xmax):
    return {x: math.exp(-lam) * lam**x / math.factorial(x) for x in range(xmax + 1)}


def pmf_pascal(k, p, xmax):
    q = 1 - p
    return {x: math.comb(k + x - 1, x) * p**k * q**x for x in range(xmax + 1)}


def pmf_hipergeometrica(N, n, p):
    Np = round(N * p)
    Nq = N - Np
    sop = {}
    for x in range(0, n + 1):
        if 0 <= x <= Np and 0 <= n - x <= Nq:
            sop[x] = math.comb(Np, x) * math.comb(Nq, n - x) / math.comb(N, n)
    return sop


# ===========================================================================
#  PROGRAMA PRINCIPAL: genera muestras y testea cada distribución
# ===========================================================================
def main():
    random.seed(2026)          # reproducibilidad
    N = 100000                # tamaño de muestra para los tests
    resultados = []
# ---------------- UNIFORME ----------------
    encabezado("UNIFORME (a=2, b=8)   [inversa + rechazo]")
    a, b = 2, 8
    ex, vx = (a + b) / 2, (b - a) ** 2 / 12
    m_inv = [uniforme_inversa(a, b) for _ in range(N)]
    m_rec = [uniforme_rechazo(a, b) for _ in range(N)]
    print("  Transformación inversa:")
    resultados.append(comparar(ex, vx, *estadisticas(m_inv)))
    print("  Método de rechazo:")
    resultados.append(comparar(ex, vx, *estadisticas(m_rec)))
    histograma_continua(m_inv)
    
    
    x_t = [a, a, b, b]
    y_t = [0, 1.0/(b-a), 1.0/(b-a), 0]
    graficar_y_guardar(m_inv, "Distribución Uniforme (a=2, b=8)", "dist_uniforme.png", True, x_t, y_t)

    # ---------------- EXPONENCIAL ----------------
    encabezado("EXPONENCIAL (EX=3)   [inversa + Von Neumann]")
    ex = 3.0
    vx = ex ** 2
    m_inv = [exponencial_inversa(ex) for _ in range(N)]
    m_vn = [exponencial_von_neumann(ex) for _ in range(N)]
    print("  Transformación inversa:")
    resultados.append(comparar(ex, vx, *estadisticas(m_inv)))
    print("  Rechazo de Von Neumann:")
    resultados.append(comparar(ex, vx, *estadisticas(m_vn)))
    histograma_continua(m_inv)
    
    
    max_exp = max(m_inv)
    x_t = [max_exp * i / 200 for i in range(201)]
    lam_val = 1.0 / ex
    y_t = [lam_val * math.exp(-lam_val * x) for x in x_t]
    graficar_y_guardar(m_inv, "Distribución Exponencial (Media=3)", "dist_exponencial.png", True, x_t, y_t)

    # ---------------- GAMMA (ERLANG) ----------------
    encabezado("GAMMA / ERLANG (EX=2, VX=1)")
    ex_g, vx_g = 2.0, 1.0
    alpha, k = gamma_parametros(ex_g, vx_g)
    print(f"  parámetros derivados: alpha={alpha:.3f}, k={k}")
    m = [gamma_erlang(alpha, k) for _ in range(N)]
    resultados.append(comparar(ex_g, vx_g, *estadisticas(m)))
    histograma_continua(m)
    
    
    max_g = max(m)
    x_t = [max_g * i / 200 for i in range(201)]
    y_t = []
    for x in x_t:
        f_val = ((alpha**k) / math.factorial(k-1)) * (x**(k-1)) * math.exp(-alpha * x) if x > 0 else 0
        y_t.append(f_val)
    graficar_y_guardar(m, f"Distribución Gamma Erlang (alpha={alpha:.2f}, k={k})", "dist_gamma.png", True, x_t, y_t)

    # ---------------- NORMAL ----------------
    encabezado("NORMAL (mu=10, sigma=2)   [límite central + Box-Muller]")
    mu, sigma = 10.0, 2.0
    ex_n, vx_n = mu, sigma ** 2
    m_lc = [normal_limite_central(mu, sigma) for _ in range(N)]
    m_bm = [normal_box_muller(mu, sigma) for _ in range(N)]
    print("  Límite central (K=12):")
    resultados.append(comparar(ex_n, vx_n, *estadisticas(m_lc)))
    print("  Box-Muller (directo):")
    resultados.append(comparar(ex_n, vx_n, *estadisticas(m_bm)))
    histograma_continua(m_bm)
    
    
    min_n, max_n = min(m_bm), max(m_bm)
    x_t = [min_n + (max_n - min_n) * i / 200 for i in range(201)]
    y_t = [(1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma)**2) for x in x_t]
    graficar_y_guardar(m_bm, "Distribución Normal (mu=10, sigma=2)", "dist_normal.png", True, x_t, y_t)

    # ---------------- PASCAL ----------------
    encabezado("PASCAL (k=3, p=0.4)")
    k_p, p_p = 3, 0.4
    q_p = 1 - p_p
    ex_p, vx_p = k_p * q_p / p_p, k_p * q_p / p_p ** 2
    m = [pascal(k_p, p_p) for _ in range(N)]
    resultados.append(comparar(ex_p, vx_p, *estadisticas(m)))
    xmax = max(m)
    histograma_discreta(m, pmf_pascal(k_p, p_p, xmax))
    
    
    pmf = pmf_pascal(k_p, p_p, xmax)
    graficar_y_guardar(m, "Distribución Pascal (k=3, p=0.4)", "dist_pascal.png", False, list(pmf.keys()), list(pmf.values()))

    # ---------------- BINOMIAL ----------------
    encabezado("BINOMIAL (n=20, p=0.3)")
    n_b, p_b = 20, 0.3
    q_b = 1 - p_b
    ex_b, vx_b = n_b * p_b, n_b * p_b * q_b
    m = [binomial(n_b, p_b) for _ in range(N)]
    resultados.append(comparar(ex_b, vx_b, *estadisticas(m)))
    histograma_discreta(m, pmf_binomial(n_b, p_b))
    
    
    pmf = pmf_binomial(n_b, p_b)
    graficar_y_guardar(m, "Distribución Binomial (n=20, p=0.3)", "dist_binomial.png", False, list(pmf.keys()), list(pmf.values()))

    # ---------------- HIPERGEOMÉTRICA ----------------
    encabezado("HIPERGEOMÉTRICA (N=50, n=10, p=0.4)")
    Nh, nh, ph = 50, 10, 0.4
    qh = 1 - ph
    ex_h = nh * ph
    vx_h = nh * ph * qh * (Nh - nh) / (Nh - 1)
    m = [hipergeometrica(Nh, nh, ph) for _ in range(N)]
    resultados.append(comparar(ex_h, vx_h, *estadisticas(m)))
    histograma_discreta(m, pmf_hipergeometrica(Nh, nh, ph))
    
    
    pmf = pmf_hipergeometrica(Nh, nh, ph)
    graficar_y_guardar(m, "Distribución Hipergeométrica (N=50, n=10, p=0.4)", "dist_hipergeometrica.png", False, list(pmf.keys()), list(pmf.values()))

    # ---------------- POISSON ----------------
    encabezado("POISSON (lambda=4)")
    lam = 4.0
    ex_po, vx_po = lam, lam
    m = [poisson(lam) for _ in range(N)]
    resultados.append(comparar(ex_po, vx_po, *estadisticas(m)))
    histograma_discreta(m, pmf_poisson(lam, max(m)))
    
    
    pmf = pmf_poisson(lam, max(m))
    graficar_y_guardar(m, "Distribución Poisson (lambda=4)", "dist_poisson.png", False, list(pmf.keys()), list(pmf.values()))

    # ---------------- EMPÍRICA DISCRETA ----------------
    encabezado("EMPÍRICA DISCRETA (tabla de Naylor, fig. pág. 135)")
    valores = list(range(1, 11))
    probs = [0.273, 0.037, 0.195, 0.009, 0.124, 0.058, 0.062, 0.151, 0.047, 0.044]
    ex_e = sum(v * pr for v, pr in zip(valores, probs))
    vx_e = sum(v * v * pr for v, pr in zip(valores, probs)) - ex_e ** 2
    m = [empirica(valores, probs) for _ in range(N)]
    resultados.append(comparar(ex_e, vx_e, *estadisticas(m)))
    histograma_discreta(m, dict(zip(valores, probs)))
    
    
    pmf = dict(zip(valores, probs))
    graficar_y_guardar(m, "Distribución Empírica Discreta", "dist_empirica.png", False, list(pmf.keys()), list(pmf.values()))

    # ---------------- RESUMEN ----------------
    encabezado("RESUMEN")
    print(f"  Pruebas que pasaron la validación: {sum(resultados)}/{len(resultados)}")


if __name__ == "__main__":
    main()







