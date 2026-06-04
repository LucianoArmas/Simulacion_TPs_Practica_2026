"""
TP2.1 - Generadores Pseudoaleatorios
---------------------------------
Integrantes:
- Tomás Lardizábal, legajo 47433
- Tomás Splivalo, legajo 51665
- Luciano Armas, legajo 47181
"""




from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra proyeccion 3D)


# ===========================================================================
# 1. GENERADORES
# ===========================================================================



class GCL:
    """Generador Congruencial Lineal (mixto).

    Recurrencia:
        X_{n+1} = (a * X_n + c) mod m
        u_n     = X_n / m

    Parámetros por defecto: los de *Numerical Recipes*, que cumplen las
    condiciones del teorema de Hull-Dobell y por lo tanto alcanzan período
    completo m = 2^32.
    """

    def __init__(self, semilla: int = 12345, a: int = 1664525, c: int = 1013904223, m: int = 2 ** 32) -> None:
        self.a = a
        self.c = c
        self.m = m
        self.semilla = semilla
        self._estado = semilla % m

    def siguiente_entero(self) -> int:
        self._estado = (self.a * self._estado + self.c) % self.m
        return self._estado

    def siguiente(self) -> float:
        """Devuelve un número pseudoaleatorio en [0, 1)."""
        return self.siguiente_entero() / self.m

    def uniformes(self, n: int) -> List[float]:
        return [self.siguiente() for _ in range(n)]

    def reiniciar(self) -> None:
        self._estado = self.semilla % self.m

    def __repr__(self) -> str:
        return f"GCL(a={self.a}, c={self.c}, m={self.m}, semilla={self.semilla})"


class RANDU(GCL):
    """GCL históricamente célebre por su mala calidad (IBM, años 60).

        X_{n+1} = 65539 * X_n mod 2^31

    Falla de forma espectacular el test espacial: todas las ternas
    (u_i, u_{i+1}, u_{i+2}) caen sobre 15 hiperplanos. Se incluye como
    contraejemplo pedagógico.
    """

    def __init__(self, semilla: int = 1) -> None:
        # La semilla debe ser impar para no degenerar.
        if semilla % 2 == 0:
            semilla += 1
        super().__init__(semilla=semilla, a=65539, c=0, m=2 ** 31)

    def __repr__(self) -> str:
        return f"RANDU(a=65539, c=0, m=2^31, semilla={self.semilla})"


class CuadradosMedios:
    """Método de los cuadrados medios de Von Neumann (1946).

    Toma una semilla de `digitos` cifras, la eleva al cuadrado, rellena
    con ceros a la izquierda hasta 2*digitos cifras y extrae las `digitos`
    cifras centrales como nuevo estado.

    Es un generador histórico de muy baja calidad: tiene períodos cortos y
    tiende a colapsar a cero. Se incluye para ilustrar esos defectos.
    """

    def __init__(self, semilla: int = 6753, digitos: int = 4) -> None:
        self.digitos = digitos
        self.semilla = semilla
        self._estado = semilla

    def siguiente_entero(self) -> int:
        cuadrado = self._estado ** 2
        # Rellenar a 2*digitos cifras con ceros a la izquierda.
        s = str(cuadrado).zfill(2 * self.digitos)
        inicio = (len(s) - self.digitos) // 2
        centro = s[inicio:inicio + self.digitos]
        self._estado = int(centro)
        return self._estado

    def siguiente(self) -> float:
        return self.siguiente_entero() / (10 ** self.digitos)

    def uniformes(self, n: int) -> List[float]:
        return [self.siguiente() for _ in range(n)]

    def reiniciar(self) -> None:
        self._estado = self.semilla

    def __repr__(self) -> str:
        return f"CuadradosMedios(semilla={self.semilla}, digitos={self.digitos})"


class PythonRandom:
    """Envoltorio sobre el generador estándar de Python (Mersenne Twister).

    Sirve como referencia de "alta calidad" para la comparación.
    """

    def __init__(self, semilla: int = 12345) -> None:
        self.semilla = semilla
        self._rng = random.Random(semilla)

    def siguiente(self) -> float:
        return self._rng.random()

    def uniformes(self, n: int) -> List[float]:
        return [self._rng.random() for _ in range(n)]

    def reiniciar(self) -> None:
        self._rng = random.Random(self.semilla)

    def __repr__(self) -> str:
        return f"PythonRandom(Mersenne Twister, semilla={self.semilla})"

# ===========================================================================
# 2. PRUEBAS ESTADISTICAS
# ===========================================================================




@dataclass
class ResultadoTest:
    nombre: str
    estadistico: float
    critico_o_p: float       # valor crítico o p-valor, según corresponda
    usa_pvalor: bool
    pasa: bool
    detalle: str = ""

    @property
    def veredicto(self) -> str:
        return "OK" if self.pasa else "ERROR"


# ---------------------------------------------------------------------------
# 1. Chi-cuadrado de uniformidad (test de frecuencias)
# ---------------------------------------------------------------------------
def test_chi_cuadrado(u: Sequence[float], k: int = 10, alpha: float = 0.05) -> ResultadoTest:
    """Particiona [0,1) en k subintervalos de igual longitud y compara la
    frecuencia observada con la esperada (n/k) mediante el estadístico
    chi-cuadrado:  X^2 = sum_i (O_i - E_i)^2 / E_i  ~ Chi^2_{k-1}.
    """
    u = np.asarray(u, dtype=float)
    n = len(u)
    observadas, _ = np.histogram(u, bins=k, range=(0.0, 1.0))
    esperada = n / k
    chi2 = float(np.sum((observadas - esperada) ** 2 / esperada))
    gl = k - 1
    p = float(stats.chi2.sf(chi2, gl))
    return ResultadoTest(
        nombre="Chi-cuadrado (uniformidad)",
        estadistico=chi2,
        critico_o_p=p,
        usa_pvalor=True,
        pasa=p >= alpha,
        detalle=f"k={k} intervalos, gl={gl}, chi2_critico={stats.chi2.ppf(1-alpha, gl):.3f}",
    )


# ---------------------------------------------------------------------------
# 2. Kolmogorov-Smirnov
# ---------------------------------------------------------------------------
def test_kolmogorov_smirnov(u: Sequence[float], alpha: float = 0.05) -> ResultadoTest:
    """Compara la distribución empírica con la U(0,1) teórica.
    D = max|F_n(x) - x|. Usa la implementación exacta de SciPy.
    """
    u = np.asarray(u, dtype=float)
    d, p = stats.kstest(u, "uniform")
    return ResultadoTest(
        nombre="Kolmogorov-Smirnov",
        estadistico=float(d),
        critico_o_p=float(p),
        usa_pvalor=True,
        pasa=p >= alpha,
        detalle=f"D_critico≈{1.36/np.sqrt(len(u)):.4f} (alpha=0.05)",
    )


# ---------------------------------------------------------------------------
# 3. Test de rachas "arriba y abajo" (independencia)
# ---------------------------------------------------------------------------
def test_rachas(u: Sequence[float], alpha: float = 0.05) -> ResultadoTest:
    """Cuenta las corridas crecientes/decrecientes de la secuencia.
    Para n grande, el número de rachas R es aproximadamente normal con:
        mu  = (2n - 1) / 3
        var = (16n - 29) / 90
    Z = (R - mu) / sqrt(var) ~ N(0,1).
    """
    u = np.asarray(u, dtype=float)
    n = len(u)
    signos = np.sign(np.diff(u))
    signos = signos[signos != 0]            # ignorar empates exactos
    if len(signos) == 0:
        return ResultadoTest("Rachas (arriba/abajo)", 0.0, 1.0, True, True, "secuencia constante")
    rachas = 1 + int(np.sum(signos[1:] != signos[:-1]))
    mu = (2 * n - 1) / 3
    var = (16 * n - 29) / 90
    z = (rachas - mu) / np.sqrt(var)
    p = float(2 * stats.norm.sf(abs(z)))
    return ResultadoTest(
        nombre="Rachas (arriba/abajo)",
        estadistico=float(z),
        critico_o_p=p,
        usa_pvalor=True,
        pasa=p >= alpha,
        detalle=f"R={rachas}, E[R]={mu:.1f}, Z_critico=±{stats.norm.ppf(1-alpha/2):.3f}",
    )


# ---------------------------------------------------------------------------
# 4. Test de póker (independencia de dígitos)
# ---------------------------------------------------------------------------
def test_poker(u: Sequence[float], alpha: float = 0.05) -> ResultadoTest:
    """Toma los 5 primeros decimales de cada número y clasifica la "mano"
    según la *cantidad de dígitos distintos* que contiene. Compara las
    frecuencias observadas con las teóricas mediante chi-cuadrado.

    Para 5 dígitos extraídos de {0,...,9} con repetición, las probabilidades
    exactas de cada número de dígitos distintos son:
        5 distintos (todos diferentes)               p = 0.3024
        4 distintos (un par)                         p = 0.5040
        3 distintos (dos pares o trío)               p = 0.1800
        <=2 distintos (full, póker o quintilla)      p = 0.0136
    (Las dos últimas manos puras se agrupan para evitar frecuencias esperadas demasiado pequeñas.)  -> 4 categorías, gl = 3.
    """
    u = np.asarray(u, dtype=float)
    categorias = {"5 dist": 0, "4 dist": 0, "3 dist": 0, "<=2 dist": 0}
    probs = {"5 dist": 0.3024, "4 dist": 0.5040, "3 dist": 0.1800, "<=2 dist": 0.0136}

    for x in u:
        digitos = [int(d) for d in f"{x:.5f}"[2:7]]   # 5 decimales
        distintos = len(set(digitos))
        if distintos == 5:
            categorias["5 dist"] += 1
        elif distintos == 4:
            categorias["4 dist"] += 1
        elif distintos == 3:
            categorias["3 dist"] += 1
        else:
            categorias["<=2 dist"] += 1

    n = len(u)
    obs = np.array([categorias[c] for c in categorias])
    esp = np.array([probs[c] * n for c in categorias])
    chi2 = float(np.sum((obs - esp) ** 2 / esp))
    gl = len(categorias) - 1
    p = float(stats.chi2.sf(chi2, gl))
    return ResultadoTest(
        nombre="Póker",
        estadistico=chi2,
        critico_o_p=p,
        usa_pvalor=True,
        pasa=p >= alpha,
        detalle=f"gl={gl}, chi2_critico={stats.chi2.ppf(1-alpha, gl):.3f}, "
                f"obs={dict(categorias)}",
    )


# ---------------------------------------------------------------------------
# 5. Test de autocorrelación a un retardo k
# ---------------------------------------------------------------------------
def test_autocorrelacion(u: Sequence[float], k: int = 1, alpha: float = 0.05) -> ResultadoTest:
    """Mide la correlación de la serie con una versión desplazada k pasos.
    Bajo independencia, el coeficiente rho_k es ~N(0, 1/(n-k)), por lo que
    Z = rho_k * sqrt(n-k) ~ N(0,1).
    """
    u = np.asarray(u, dtype=float)
    n = len(u)
    a = u[:-k] - u.mean()
    b = u[k:] - u.mean()
    rho = float(np.sum(a * b) / np.sum((u - u.mean()) ** 2))
    z = rho * np.sqrt(n - k)
    p = float(2 * stats.norm.sf(abs(z)))
    return ResultadoTest(
        nombre=f"Autocorrelación (lag={k})",
        estadistico=z,
        critico_o_p=p,
        usa_pvalor=True,
        pasa=p >= alpha,
        detalle=f"rho_{k}={rho:.5f}, Z_critico=±{stats.norm.ppf(1-alpha/2):.3f}",
    )


# ---------------------------------------------------------------------------
# Batería completa
# ---------------------------------------------------------------------------
def bateria_completa(u: Sequence[float], alpha: float = 0.05) -> List[ResultadoTest]:
    """Ejecuta las cinco pruebas sobre la secuencia u."""
    return [
        test_chi_cuadrado(u, k=10, alpha=alpha),
        test_kolmogorov_smirnov(u, alpha=alpha),
        test_rachas(u, alpha=alpha),
        test_poker(u, alpha=alpha),
        test_autocorrelacion(u, k=1, alpha=alpha),
    ]

# ===========================================================================
# 3. ORQUESTACION, TABLAS Y FIGURAS
# ===========================================================================




# --------------------------------------------------------------------------- #
N = 10000          # tamaño de muestra para las pruebas estadísticas
ALPHA = 0.05        # nivel de significación
DIR_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
os.makedirs(DIR_FIG, exist_ok=True)

# Paleta sobria para las figuras.
COLOR = "#1f375f"
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})


def fig_path(nombre: str) -> str:
    return os.path.join(DIR_FIG, nombre)


# --------------------------------------------------------------------------- #
def construir_generadores() -> Dict[str, object]:
    return {
        "GCL": GCL(semilla=12345),
        "RANDU": RANDU(semilla=1),
        "Cuadrados Medios": CuadradosMedios(semilla=6753, digitos=4),
        "Python (Mersenne)": PythonRandom(semilla=12345),
    }


def correr_pruebas() -> Dict[str, Dict[str, List[ResultadoTest]]]:
    resultados = {}
    muestras = {}
    for nombre, gen in construir_generadores().items():
        gen.reiniciar()
        u = gen.uniformes(N)
        muestras[nombre] = np.asarray(u)
        resultados[nombre] = bateria_completa(u, alpha=ALPHA)
    return resultados, muestras


# --------------------------------------------------------------------------- #
def imprimir_y_guardar(resultados, muestras) -> None:
    nombres_test = [r.nombre for r in next(iter(resultados.values()))]
    lineas = []

    def out(s=""):
        print(s)
        lineas.append(s)

    out("=" * 78)
    out(f"TP 2.1 - Pruebas de generadores pseudoaleatorios  (n={N}, alpha={ALPHA})")
    out("=" * 78)

    # Tabla resumen veredictos.
    out("\nRESUMEN DE VEREDICTOS (OK = no se rechaza H0)\n")
    cab = f"{'Generador':<20}" + "".join(f"{t.split(' ')[0][:9]:>11}" for t in nombres_test)
    out(cab)
    out("-" * len(cab))
    for gen, res in resultados.items():
        fila = f"{gen:<20}" + "".join(f"{r.veredicto:>11}" for r in res)
        out(fila)

    # Detalle por generador.
    for gen, res in resultados.items():
        out(f"\n{'-'*78}\n{gen}\n{'-'*78}")
        u = muestras[gen]
        out(f"   media = {u.mean():.5f}   (esperada 0.5)   "
            f"varianza = {u.var():.5f}   (esperada {1/12:.5f})")
        for r in res:
            metric = "p-valor" if r.usa_pvalor else "crítico"
            out(f"   {r.nombre:<28} estad={r.estadistico: .4f}  "
                f"{metric}={r.critico_o_p:.4f}  -> {r.veredicto}")
            if r.detalle:
                out(f"        {r.detalle}")

    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    out(f"\n[resultados guardados en {ruta}]")


# --------------------------------------------------------------------------- #
def figura_histogramas(muestras) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, (nombre, u) in zip(axes.ravel(), muestras.items()):
        ax.hist(u, bins=20, range=(0, 1), color=COLOR, edgecolor="white", alpha=0.85)
        ax.axhline(len(u) / 20, color="crimson", ls="--", lw=1.2, label="frecuencia esperada")
        ax.set_title(nombre)
        ax.set_xlabel("valor")
        ax.set_ylabel("frecuencia")
        ax.legend(fontsize=7)
    fig.suptitle("Histogramas de uniformidad (20 intervalos)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path("histogramas.png"))
    plt.close(fig)


def figura_dispersion(muestras) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    m = 2000
    for ax, (nombre, u) in zip(axes.ravel(), muestras.items()):
        x, y = u[:-1][:m], u[1:][:m]
        ax.scatter(x, y, s=4, color=COLOR, alpha=0.5)
        ax.set_title(nombre)
        ax.set_xlabel("$u_i$")
        ax.set_ylabel("$u_{i+1}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    fig.suptitle("Diagramas de dispersión de retardo ($u_i$ vs $u_{i+1}$)", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path("dispersion.png"))
    plt.close(fig)


def figura_randu_3d_old() -> None:
    """La célebre estructura reticular de RANDU. Como los planos son difíciles
    de ver en una proyección 3D arbitraria, además se proyectan las ternas
    sobre el vector normal teórico a los planos, N = (9, -6, 1), que surge de
    la relación  x_{i+2} = 6 x_{i+1} - 9 x_i  (mod 2^31). Esa proyección
    revela las 15 capas discretas de forma inequívoca.
    """
    gen = RANDU(semilla=1)
    u = np.asarray(gen.uniformes(12000))
    x, y, z = u[0::3], u[1::3], u[2::3]
    n = min(len(x), len(y), len(z))
    x, y, z = x[:n], y[:n], z[:n]
    P = np.column_stack([x, y, z])

    # Vector normal a los planos y dos vectores que generan el plano.
    N = np.array([9.0, -6.0, 1.0]); N /= np.linalg.norm(N)
    v1 = np.array([1.0, 1.0, -3.0]); v1 -= (v1 @ N) * N; v1 /= np.linalg.norm(v1)
    v2 = np.cross(N, v1)
    coord_normal = P @ N           # cuantizada -> bandas
    coord_plano = P @ v1

    fig = plt.figure(figsize=(9.5, 4.2))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(x, y, z, s=3, color=COLOR, alpha=0.5)
    ax1.view_init(elev=22, azim=-60)
    ax1.set_title("Ternas en el cubo unitario\n(aparentan ser aleatorias)")
    ax1.set_xlabel("$u_i$"); ax1.set_ylabel("$u_{i+1}$"); ax1.set_zlabel("$u_{i+2}$")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(coord_normal, coord_plano, s=3, color="crimson", alpha=0.45)
    ax2.set_title("Proyección sobre el normal $(9,-6,1)$\n(15 planos paralelos)")
    ax2.set_xlabel("proyección $\\cdot\\,(9,-6,1)/\\|\\cdot\\|$")
    ax2.set_ylabel("proyección en el plano")

    fig.suptitle("Estructura reticular de RANDU: el defecto que los tests 1D no detectan", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path("randu_3d.png"))
    plt.close(fig)

def figura_randu_3d_2(u: Sequence[float]) -> None:
    """Demuestra la estructura reticular de RANDU usando los datos ingresados.
    Se agrupan los números en ternas (x, y, z) y se colorean según el plano
    al que pertenecen. Además, se dibuja el vector normal que delata la falla.
    """
    u = np.asarray(u)
    x, y, z = u[0::3], u[1::3], u[2::3]
    
    n = min(len(x), len(y), len(z))
    x, y, z = x[:n], y[:n], z[:n]

    ecuacion = 9 * x - 6 * y + z
    planos = np.round(ecuacion) 

    fig = plt.figure(figsize=(11, 5))

    # --- Gráfico Izquierdo: El cubo 3D coloreado ---
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(x, y, z, c=planos, cmap="tab20", s=6, alpha=0.6, edgecolor="none")
    
    # ---------------------------------------------------------
    # NUEVO: Dibujar el vector normal (9, -6, 1)
    # ---------------------------------------------------------
    # 1. Definimos el vector y calculamos su longitud para normalizarlo
    N = np.array([9.0, -6.0, 1.0])
    N_norm = N / np.linalg.norm(N)
    
    # 2. Lo posicionamos en el centro del cubo (0.5, 0.5, 0.5)
    origen = np.array([0.5, 0.5, 0.5])
    
    # 3. Dibujamos la flecha (multiplicamos N_norm por 0.6 para que tenga buen tamaño visual)
    ax1.quiver(origen[0], origen[1], origen[2], 
               N_norm[0] * 0.6, N_norm[1] * 0.6, N_norm[2] * 0.6, color="crimson", linewidth=3, arrow_length_ratio=0.15)
    
    # 4. Le ponemos una etiqueta justo en la punta de la flecha
    ax1.text(origen[0] + N_norm[0]*0.6, origen[1] + N_norm[1]*0.6, origen[2] + N_norm[2]*1, " Vector Normal\n (9, -6, 1)", color="crimson", fontsize=9, fontweight="bold")
    # ---------------------------------------------------------
    # OLD:
    # ax1.view_init(elev=15, azim=-62) 
    # ax1.view_init(elev=5, azim=55)  # CLAVE
    # NEW:
    ax1.view_init(elev=5, azim=55) 
    ax1.set_title("Ternas en el cubo unitario\n(Coloreadas según el plano al que pertenecen)")
    ax1.set_xlabel("$u_i$"); ax1.set_ylabel("$u_{i+1}$"); ax1.set_zlabel("$u_{i+2}$")

    # --- Gráfico Derecho: Vista lateral matemática plana ---
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(x, ecuacion, c=planos, cmap="tab20", s=4, alpha=0.7, edgecolor="none")
    ax2.set_title("Resultado de $9u_i - 6u_{i+1} + u_{i+2}$\n(Demostración de los 15 niveles)")
    ax2.set_xlabel("Valor de $u_i$ (eje visual de dispersión)")
    ax2.set_ylabel("Nivel del plano (Entero)")
    
    for i in range(-6, 11):
        ax2.axhline(i, color="gray", lw=0.5, alpha=0.3, zorder=0)

    fig.suptitle("Estructura reticular de RANDU: Los puntos están atrapados en 15 planos paralelos", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path("randu_3d.png"))
    plt.close(fig)

def figura_randu_3d(u: Sequence[float]) -> None:
    """Demuestra la estructura reticular de RANDU usando los datos ingresados.
    Muestra dos vistas 3D del cubo desde distintos ángulos y la proyección 2D.
    """
    u = np.asarray(u)
    x, y, z = u[0::3], u[1::3], u[2::3]
    
    n = min(len(x), len(y), len(z))
    x, y, z = x[:n], y[:n], z[:n]

    ecuacion = 9 * x - 6 * y + z
    planos = np.round(ecuacion) 

    # Agrandamos la figura para que entren 3 columnas cómodamente
    fig = plt.figure(figsize=(16, 5))

    # Definimos los ángulos que queremos mostrar: (elevación, azimut)
    # El primero es el que encontraste vos, el segundo es otro ángulo útil.
    angulos_vistas = [
        (15, -62),
        (5, -20),
        (5, 30),
        (5, 55)
    ]

    for i, (elev, azim) in enumerate(angulos_vistas):

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            x, y, z,
            c=planos,
            cmap="tab20",
            s=6,
            alpha=0.6,
            edgecolor="none"
        )

        N = np.array([9.0, -6.0, 1.0])
        N_norm = N / np.linalg.norm(N)
        origen = np.array([0.5, 0.5, 0.5])

        ax.quiver(
            origen[0], origen[1], origen[2],
            N_norm[0] * 0.6, N_norm[1] * 0.6, N_norm[2] * 0.6,
            color="crimson",
            linewidth=3,
            arrow_length_ratio=0.15
        )
        
        ax.text(origen[0] + N_norm[0]*0.6, origen[1] + N_norm[1]*0.6, origen[2] + N_norm[2]*1, " Vector Normal\n (9, -6, 1)", color="crimson", fontsize=9, fontweight="bold")

        ax.view_init(elev=elev, azim=azim)

        ax.set_title(f"Ternas en cubo unitario \n(elev={elev}°, azim={azim}°)")
        ax.set_xlabel("$u_i$")
        ax.set_ylabel("$u_{i+1}$")
        ax.set_zlabel("$u_{i+2}$")

        fig.tight_layout()

        fig.savefig(
            fig_path(f"randu_3d_vista_{i+1}.png"),
            bbox_inches="tight"
        )

        plt.close(fig)
    
    # --- Gráfico 3: Vista lateral matemática plana (2D) ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        x,
        ecuacion,
        c=planos,
        cmap="tab20",
        s=4,
        alpha=0.7,
        edgecolor="none"
    )

    ax.set_title(
        "Resultado de $9u_i - 6u_{i+1} + u_{i+2}$\n"
        "(Demostración de los 15 niveles)"
    )

    ax.set_xlabel("Valor de $u_i$ (eje visual)")
    ax.set_ylabel("Nivel del plano (Entero)")

    for i in range(-6, 11):
        ax.axhline(i, color="gray", lw=0.5, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        fig_path("randu_15_planos.png"),
        bbox_inches="tight"
    )

    plt.close(fig)



# --------------------------------------------------------------------------- #
def main() -> None:
    resultados, muestras = correr_pruebas()
    imprimir_y_guardar(resultados, muestras)
    figura_histogramas(muestras)
    figura_dispersion(muestras)
    figura_randu_3d(muestras["RANDU"])
    print(f"\n[figuras guardadas en {os.path.abspath(DIR_FIG)}]")

if __name__ == "__main__":
    main()
















