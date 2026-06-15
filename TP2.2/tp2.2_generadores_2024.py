#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP 2.2 - GENERADORES DE NÚMEROS PSEUDOALEATORIOS DE DISTINTAS DISTRIBUCIONES
============================================================================
UTN - FRRO | Simulación

Implementa generadores de variables aleatorias para las distribuciones pedidas
en la consigna, siguiendo los métodos del Cap. 4 de:
    Naylor, T. H. "Técnicas de Simulación en Computadoras", 1982.

Métodos cubiertos (entre paréntesis la ecuación de Naylor):
    UNIFORME        -> transformación inversa (4-23) y método de rechazo (4-12..4-14)
    EXPONENCIAL     -> transformación inversa (4-30) y rechazo de Von Neumann (4-31)
    GAMMA (Erlang)  -> suma de k exponenciales / inversa (4-53, 4-54)
    NORMAL          -> método del límite central K=12 (4-75) y directo Box-Muller (4-81/82)
    PASCAL          -> suma de k geométricas (4-133)
    BINOMIAL        -> reproducción de ensayos de Bernoulli (4-142, 4-143)
    HIPERGEOMÉTRICA -> muestreo sin reemplazo (4-147, 4-148)
    POISSON         -> producto de uniformes vs e^(-lambda) (4-153)
    EMPÍRICA DISC.  -> transformación inversa sobre la acumulada (4-154)

Un único número uniforme U(0,1) es la fuente base de TODO (función u()).
En el TP anterior se construyó y testeó ese generador; aquí simplemente se usa.
"""

import math
import random

# ===========================================================================
#  FUENTE UNIFORME BASE  U(0,1)
# ===========================================================================
def u():
    """Devuelve un número pseudoaleatorio uniforme en (0,1).

    Es la única primitiva de aleatoriedad: el resto de los generadores se
    construyen transformando este valor. En el TP anterior se validó un
    generador propio; aquí se usa random.random() como esa fuente ya testeada.
    """
    return random.random()


# ===========================================================================
#  DISTRIBUCIONES CONTINUAS
# ===========================================================================

# ---------------------------- UNIFORME (a, b) ------------------------------
def uniforme_inversa(a, b):
    """Transformación inversa.  F(x) = (x-a)/(b-a)  =>  x = a + (b-a)*r   (4-23)."""
    return a + (b - a) * u()


def uniforme_rechazo(a, b):
    """Método de rechazo (4-12..4-14).

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
    """Transformación inversa.  r = e^(-alpha*x)  =>  x = -EX*ln(r)   (4-30).
    ex = EX = media = 1/alpha."""
    return -ex * math.log(u())


def exponencial_von_neumann(ex=1.0):
    """Método de rechazo de Von Neumann (4-31), sin usar logaritmo.

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

    x = -(1/alpha) * ln( PROD r_i )    (4-53 / 4-54).
    EX = k/alpha ,  VX = k/alpha^2.
    """
    prod = 1.0
    for _ in range(k):
        prod *= u()
    return -math.log(prod) / alpha


def gamma_parametros(ex, vx):
    """Deriva (alpha, k) a partir de la media y la varianza  (4-51, 4-52)."""
    alpha = ex / vx
    k = round(ex * ex / vx)
    return alpha, k


# ------------------------------ NORMAL (mu, sigma) -------------------------
def normal_limite_central(mu, sigma):
    """Método del límite central con K=12  (4-75).

    x = sigma * (SUM_{i=1}^{12} r_i - 6) + mu.
    Con K=12 se evita la multiplicación por sqrt(12/K) y queda truncada a +/-6 sigma.
    """
    s = 0.0
    for _ in range(12):
        s += u()
    return sigma * (s - 6.0) + mu


def normal_box_muller(mu, sigma):
    """Procedimiento directo de Box-Muller  (4-81 / 4-82), resultado exacto."""
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

    Es la SUMA de k variables geométricas, cada una generada por inversa
    (4-125):  x_j = floor( ln(r_j) / ln(q) ).  Cuenta el número de fracasos
    antes del k-ésimo éxito.   EX = k*q/p ,  VX = k*q/p^2   (q = 1 - p).

    Nota: la forma compacta de Naylor (4-133) escribe x = floor(ln(PROD r_i)/ln q),
    pero floor(suma) != suma(floor); esa versión truncada NO reproduce la
    binomial negativa (sesga la media). Por eso aquí se suman k geométricas
    truncadas individualmente, que es lo correcto.
    """
    q = 1.0 - p
    lnq = math.log(q)
    x = 0
    for _ in range(k):
        x += int(math.log(u()) / lnq)     # geométrica (4-125)
    return x


# ----------------------------- BINOMIAL (n, p) -----------------------------
def binomial(n, p):
    """Reproducción de n ensayos de Bernoulli  (4-142, 4-143).

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
    """Muestreo SIN reemplazo  (4-147, 4-148).

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
    """Producto de uniformes comparado con e^(-lambda)  (4-153).

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
    """Transformación inversa sobre la función acumulada  (4-154).

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

    # ---------------- GAMMA (ERLANG) ----------------
    encabezado("GAMMA / ERLANG (EX=2, VX=1)")
    ex, vx = 2.0, 1.0
    alpha, k = gamma_parametros(ex, vx)
    print(f"  parámetros derivados: alpha={alpha:.3f}, k={k}")
    m = [gamma_erlang(alpha, k) for _ in range(N)]
    resultados.append(comparar(ex, vx, *estadisticas(m)))
    histograma_continua(m)

    # ---------------- NORMAL ----------------
    encabezado("NORMAL (mu=10, sigma=2)   [límite central + Box-Muller]")
    mu, sigma = 10.0, 2.0
    ex, vx = mu, sigma ** 2
    m_lc = [normal_limite_central(mu, sigma) for _ in range(N)]
    m_bm = [normal_box_muller(mu, sigma) for _ in range(N)]
    print("  Límite central (K=12):")
    resultados.append(comparar(ex, vx, *estadisticas(m_lc)))
    print("  Box-Muller (directo):")
    resultados.append(comparar(ex, vx, *estadisticas(m_bm)))
    histograma_continua(m_bm)

    # ---------------- PASCAL ----------------
    encabezado("PASCAL (k=3, p=0.4)")
    k, p = 3, 0.4
    q = 1 - p
    ex, vx = k * q / p, k * q / p ** 2
    m = [pascal(k, p) for _ in range(N)]
    resultados.append(comparar(ex, vx, *estadisticas(m)))
    xmax = max(m)
    histograma_discreta(m, pmf_pascal(k, p, xmax))

    # ---------------- BINOMIAL ----------------
    encabezado("BINOMIAL (n=20, p=0.3)")
    n, p = 20, 0.3
    q = 1 - p
    ex, vx = n * p, n * p * q
    m = [binomial(n, p) for _ in range(N)]
    resultados.append(comparar(ex, vx, *estadisticas(m)))
    histograma_discreta(m, pmf_binomial(n, p))

    # ---------------- HIPERGEOMÉTRICA ----------------
    encabezado("HIPERGEOMÉTRICA (N=50, n=10, p=0.4)")
    Nh, nh, ph = 50, 10, 0.4
    qh = 1 - ph
    ex = nh * ph
    vx = nh * ph * qh * (Nh - nh) / (Nh - 1)
    m = [hipergeometrica(Nh, nh, ph) for _ in range(N)]
    resultados.append(comparar(ex, vx, *estadisticas(m)))
    histograma_discreta(m, pmf_hipergeometrica(Nh, nh, ph))

    # ---------------- POISSON ----------------
    encabezado("POISSON (lambda=4)")
    lam = 4.0
    ex, vx = lam, lam
    m = [poisson(lam) for _ in range(N)]
    resultados.append(comparar(ex, vx, *estadisticas(m)))
    histograma_discreta(m, pmf_poisson(lam, max(m)))

    # ---------------- EMPÍRICA DISCRETA ----------------
    encabezado("EMPÍRICA DISCRETA (tabla de Naylor, fig. pág. 135)")
    valores = list(range(1, 11))
    probs = [0.273, 0.037, 0.195, 0.009, 0.124, 0.058, 0.062, 0.151, 0.047, 0.044]
    ex = sum(v * pr for v, pr in zip(valores, probs))
    vx = sum(v * v * pr for v, pr in zip(valores, probs)) - ex ** 2
    m = [empirica(valores, probs) for _ in range(N)]
    resultados.append(comparar(ex, vx, *estadisticas(m)))
    histograma_discreta(m, dict(zip(valores, probs)))

    # ---------------- RESUMEN ----------------
    encabezado("RESUMEN")
    print(f"  Pruebas que pasaron la validación: {sum(resultados)}/{len(resultados)}")


if __name__ == "__main__":
    main()
