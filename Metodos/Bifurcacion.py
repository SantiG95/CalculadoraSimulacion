"""
╔══════════════════════════════════════════════════════════════╗
║         ANÁLISIS DE BIFURCACIÓN - Sistema Dinámico           ║
║     Diagrama de bifurcación, puntos críticos y estabilidad   ║
╚══════════════════════════════════════════════════════════════╝

Uso:
    python bifurcacion_analisis.py

El script permite ingresar una función f(x, r) y analiza:
  - Puntos de equilibrio (f(x, r) = 0)
  - Estabilidad local (df/dx en cada equilibrio)
  - Diagrama de bifurcación
  - Detección de bifurcaciones silla-nodo, transcrítica, horquilla y Hopf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import brentq, fsolve
from scipy.differentiate import derivative as _sp_deriv

def derivative(func, x0, dx=1e-6, **kwargs):
    """Derivada numérica compatible con scipy moderno y antiguo."""
    return (func(x0 + dx) - func(x0 - dx)) / (2 * dx)
import warnings
import sys
import re

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Paleta de colores
# ─────────────────────────────────────────────
COLORS = {
    "bg":        "#0d0d1a",
    "panel":     "#13132b",
    "stable":    "#00e5ff",
    "unstable":  "#ff4081",
    "neutral":   "#ffd740",
    "bifurc":    "#69ff47",
    "grid":      "#1e1e3f",
    "text":      "#e0e0ff",
    "accent":    "#bb86fc",
}

plt.rcParams.update({
    "figure.facecolor":  COLORS["bg"],
    "axes.facecolor":    COLORS["panel"],
    "axes.edgecolor":    COLORS["grid"],
    "axes.labelcolor":   COLORS["text"],
    "xtick.color":       COLORS["text"],
    "ytick.color":       COLORS["text"],
    "text.color":        COLORS["text"],
    "grid.color":        COLORS["grid"],
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    10,
})


# ══════════════════════════════════════════════
#  PARSER DE FUNCIÓN SEGURO
# ══════════════════════════════════════════════

SAFE_NAMESPACE = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "sinh": np.sinh, "cosh": np.cosh,
    "tanh": np.tanh, "arcsin": np.arcsin, "arccos": np.arccos,
    "arctan": np.arctan, "pi": np.pi, "e": np.e,
    "sign": np.sign, "ceil": np.ceil, "floor": np.floor,
}


def parse_function(expr_str: str):
    """Convierte una cadena en función f(x, r) segura."""
    # Reemplazos de notación amigable
    expr = expr_str.strip()
    expr = expr.replace("^", "**")
    expr = re.sub(r"(\d)(x)", r"\1*\2", expr)   # 2x → 2*x
    expr = re.sub(r"(\d)(r)", r"\1*\2", expr)   # 2r → 2*r
    expr = re.sub(r"(x)(r)", r"\1*\2", expr)
    expr = re.sub(r"(\))(x|r|\d)", r"\1*\2", expr)
    expr = re.sub(r"(x|r)(\()", r"\1*\2", expr)

    def f(x, r):
        ns = {**SAFE_NAMESPACE, "x": x, "r": r}
        try:
            return eval(expr, {"__builtins__": {}}, ns)  # noqa: S307
        except Exception:
            return np.nan

    return f, expr


# ══════════════════════════════════════════════
#  MOTOR DE ANÁLISIS
# ══════════════════════════════════════════════

def find_equilibria(f, r_val: float, x_range: tuple, n_scan: int = 800):
    """Encuentra raíces de f(x, r) = 0 para un r fijo."""
    x_vals = np.linspace(x_range[0], x_range[1], n_scan)
    try:
        fx = np.array([f(xi, r_val) for xi in x_vals], dtype=float)
    except Exception:
        return []

    roots = []
    for i in range(len(fx) - 1):
        if np.isnan(fx[i]) or np.isnan(fx[i + 1]):
            continue
        if fx[i] * fx[i + 1] < 0:
            try:
                root = brentq(lambda x: f(x, r_val), x_vals[i], x_vals[i + 1], xtol=1e-9)
                if not any(abs(root - r) < 1e-6 for r in roots):
                    roots.append(root)
            except Exception:
                pass
    return roots


def stability(f, x_eq: float, r_val: float, dx: float = 1e-6) -> str:
    """Clasifica estabilidad: 'stable', 'unstable', 'neutral'."""
    try:
        df = derivative(lambda x: f(x, r_val), x_eq, dx=dx)
        if df < -1e-8:
            return "stable"
        elif df > 1e-8:
            return "unstable"
        else:
            return "neutral"
    except Exception:
        return "unknown"


def build_bifurcation_data(f, r_range: tuple, x_range: tuple,
                            n_r: int = 600, n_scan: int = 500):
    """Construye los datos para el diagrama de bifurcación."""
    r_vals = np.linspace(r_range[0], r_range[1], n_r)
    stable_pts, unstable_pts, neutral_pts = [], [], []

    for r in r_vals:
        eqs = find_equilibria(f, r, x_range, n_scan)
        for x_eq in eqs:
            stab = stability(f, x_eq, r)
            if stab == "stable":
                stable_pts.append((r, x_eq))
            elif stab == "unstable":
                unstable_pts.append((r, x_eq))
            else:
                neutral_pts.append((r, x_eq))

    return (np.array(stable_pts),
            np.array(unstable_pts),
            np.array(neutral_pts))


def detect_bifurcation_points(f, r_range: tuple, x_range: tuple,
                               n_r: int = 400, n_scan: int = 400):
    """Detecta cambios en número de equilibrios → puntos de bifurcación."""
    r_vals = np.linspace(r_range[0], r_range[1], n_r)
    bif_points = []
    prev_count = None

    for r in r_vals:
        eqs = find_equilibria(f, r, x_range, n_scan)
        count = len(eqs)
        if prev_count is not None and count != prev_count:
            bif_points.append((r, eqs))
        prev_count = count

    return bif_points


# ══════════════════════════════════════════════
#  ANÁLISIS DE ESTABILIDAD LOCAL DETALLADO
# ══════════════════════════════════════════════

def full_stability_report(f, r_sample: float, x_range: tuple):
    """Genera reporte de texto para un valor r dado."""
    eqs = find_equilibria(f, r_sample, x_range)
    lines = []
    lines.append(f"\n{'─'*52}")
    lines.append(f"  Análisis en r = {r_sample:.4f}")
    lines.append(f"{'─'*52}")
    if not eqs:
        lines.append("  ⚠  No se encontraron puntos de equilibrio.")
    for xeq in eqs:
        stab = stability(f, xeq, r_sample)
        try:
            dfdx = derivative(lambda x: f(x, r_sample), xeq, dx=1e-6)
        except Exception:
            dfdx = float("nan")

        icon = {"stable": "●  ESTABLE", "unstable": "○  INESTABLE",
                "neutral": "◐  NEUTRAL / SEMI-ESTABLE"}.get(stab, "?")
        lines.append(f"\n  x* = {xeq:+.6f}")
        lines.append(f"  Estado     : {icon}")
        lines.append(f"  f'(x*, r)  : {dfdx:+.6f}  {'< 0 → atractor' if dfdx < 0 else '> 0 → repulsor' if dfdx > 0 else '= 0 → indeterminado'}")
    lines.append(f"{'─'*52}\n")
    return "\n".join(lines)


# ══════════════════════════════════════════════
#  VISUALIZACIÓN
# ══════════════════════════════════════════════

def plot_phase_portrait(ax, f, r_val: float, x_range: tuple):
    """Retrato de fase: f(x) vs x con flechas de flujo."""
    xs = np.linspace(x_range[0], x_range[1], 500)
    ys = np.array([f(xi, r_val) for xi in xs], dtype=float)

    ax.axhline(0, color=COLORS["grid"], lw=1.0, zorder=1)
    ax.plot(xs, ys, color=COLORS["accent"], lw=2, zorder=3)
    ax.fill_between(xs, ys, 0, where=(ys > 0), alpha=0.12,
                    color=COLORS["stable"], zorder=2)
    ax.fill_between(xs, ys, 0, where=(ys < 0), alpha=0.12,
                    color=COLORS["unstable"], zorder=2)

    # Flechas de flujo
    arrow_xs = np.linspace(x_range[0] * 0.85, x_range[1] * 0.85, 14)
    for ax_val in arrow_xs:
        fv = f(ax_val, r_val)
        if np.isnan(fv) or abs(fv) < 1e-10:
            continue
        direction = 1 if fv > 0 else -1
        ax.annotate("", xy=(ax_val + direction * 0.04 * (x_range[1] - x_range[0]), 0),
                    xytext=(ax_val, 0),
                    arrowprops=dict(arrowstyle="->", color=COLORS["text"],
                                   lw=0.8, alpha=0.5))

    # Equilibrios
    eqs = find_equilibria(f, r_val, x_range)
    for xeq in eqs:
        stab = stability(f, xeq, r_val)
        color = COLORS["stable"] if stab == "stable" else (
            COLORS["unstable"] if stab == "unstable" else COLORS["neutral"])
        marker = "o" if stab == "stable" else ("s" if stab == "unstable" else "^")
        ax.scatter([xeq], [0], color=color, s=90, zorder=6,
                   marker=marker, edgecolors="white", linewidths=0.7)
        ax.axvline(xeq, color=color, lw=0.6, ls="--", alpha=0.4)

    ax.set_title(f"Retrato de fase  (r = {r_val:.3f})", color=COLORS["accent"])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x, r)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(np.nanpercentile(ys, 1) * 1.3, np.nanpercentile(ys, 99) * 1.3)


def plot_bifurcation(ax, stable, unstable, neutral, bif_pts, r_range):
    """Diagrama de bifurcación principal."""
    ax.set_title("Diagrama de Bifurcación", color=COLORS["accent"])
    ax.set_xlabel("Parámetro  r")
    ax.set_ylabel("Equilibrios  x*")
    ax.grid(True, alpha=0.3)

    if len(stable) > 0:
        ax.scatter(stable[:, 0], stable[:, 1], s=2.5, color=COLORS["stable"],
                   alpha=0.8, label="Estable", rasterized=True)
    if len(unstable) > 0:
        ax.scatter(unstable[:, 0], unstable[:, 1], s=2.5,
                   color=COLORS["unstable"], alpha=0.8, label="Inestable",
                   rasterized=True)
    if len(neutral) > 0:
        ax.scatter(neutral[:, 0], neutral[:, 1], s=2.5, color=COLORS["neutral"],
                   alpha=0.8, label="Neutral", rasterized=True)

    # Puntos de bifurcación
    for r_b, eqs_b in bif_pts:
        ax.axvline(r_b, color=COLORS["bifurc"], lw=1.0, ls="--", alpha=0.7)
        for xb in eqs_b:
            ax.scatter([r_b], [xb], color=COLORS["bifurc"],
                       s=60, zorder=7, marker="D",
                       edgecolors="white", linewidths=0.6)

    ax.legend(loc="upper right", fontsize=8,
              facecolor=COLORS["bg"], edgecolor=COLORS["grid"])

    # Eje r = 0 si está en rango
    if r_range[0] < 0 < r_range[1]:
        ax.axvline(0, color=COLORS["text"], lw=0.5, alpha=0.3)


def plot_eigenvalue_map(ax, f, r_range: tuple, x_range: tuple, n_r: int = 400):
    """Mapa del autovalor df/dx en el espacio (r, x*)."""
    r_vals = np.linspace(r_range[0], r_range[1], n_r)
    rs, xs, lambdas = [], [], []

    for r in r_vals:
        eqs = find_equilibria(f, r, x_range)
        for xeq in eqs:
            try:
                lam = derivative(lambda x: f(x, r), xeq, dx=1e-6)
                rs.append(r)
                xs.append(xeq)
                lambdas.append(lam)
            except Exception:
                pass

    if not rs:
        ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes,
                ha="center", va="center", color=COLORS["text"])
        return

    rs = np.array(rs)
    xs = np.array(xs)
    lambdas = np.array(lambdas)

    lim = np.percentile(np.abs(lambdas[np.isfinite(lambdas)]), 95) + 0.1
    sc = ax.scatter(rs, xs, c=lambdas, cmap="RdBu_r",
                    vmin=-lim, vmax=lim, s=3, alpha=0.85, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("df/dx  (autovalor)", color=COLORS["text"])
    cbar.ax.yaxis.set_tick_params(color=COLORS["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["text"])

    ax.set_title("Mapa de Autovalores", color=COLORS["accent"])
    ax.set_xlabel("Parámetro  r")
    ax.set_ylabel("x*")
    ax.grid(True, alpha=0.3)


def plot_number_of_eq(ax, f, r_range: tuple, x_range: tuple, n_r: int = 500):
    """Número de equilibrios en función de r."""
    r_vals = np.linspace(r_range[0], r_range[1], n_r)
    counts = [len(find_equilibria(f, r, x_range)) for r in r_vals]

    ax.step(r_vals, counts, where="mid", color=COLORS["accent"], lw=2)
    ax.fill_between(r_vals, counts, step="mid",
                    color=COLORS["accent"], alpha=0.15)
    ax.set_title("Número de Equilibrios vs r", color=COLORS["accent"])
    ax.set_xlabel("Parámetro  r")
    ax.set_ylabel("# equilibrios")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════
#  INTERFAZ DE USUARIO (CONSOLA)
# ══════════════════════════════════════════════

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║         A N Á L I S I S   D E   B I F U R C A C I Ó N       ║
╚══════════════════════════════════════════════════════════════╝

  Variables : x  (estado)   r  (parámetro de bifurcación)
  Operadores: +  -  *  /  **  ^
  Funciones : sin  cos  exp  log  sqrt  abs  tanh  ...
  Constantes: pi  e

  Ejemplos de sistemas clásicos:
    ① Silla-nodo     :  r + x**2          (o  r + x^2)
    ② Transcrítica   :  r*x - x**2
    ③ Horquilla pitchfork (supercrítica): r*x - x**3
    ④ Horquilla (subcrítica): r*x + x**3
    ⑤ Logística      :  r*x*(1 - x)
    ⑥ Van der Pol    :  r*x - x**3/3      (sistema reducido)
    ⑦ Personalizada  :  cualquier f(x, r)
"""


def get_float(prompt: str, default: float) -> float:
    s = input(f"  {prompt} [{default}]: ").strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        print(f"  ⚠  Entrada inválida, usando {default}")
        return default


def get_int(prompt: str, default: int) -> int:
    s = input(f"  {prompt} [{default}]: ").strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        print(f"  ⚠  Entrada inválida, usando {default}")
        return default


def main():
    print(BANNER)

    # ── Ingreso de función ───────────────────
    while True:
        expr = input("  Ingresa f(x, r) = ").strip()
        if not expr:
            print("  ⚠  La expresión no puede estar vacía.\n")
            continue
        f, parsed = parse_function(expr)
        # Prueba rápida
        try:
            val = f(0.0, 0.0)
            if np.isnan(val) and "log" not in expr and "sqrt" not in expr:
                print("  ⚠  La función devolvió NaN en (0,0). Verifica la expresión.\n")
                continue
        except Exception as e:
            print(f"  ⚠  Error al evaluar: {e}\n")
            continue
        print(f"\n  ✔  Función reconocida: f(x, r) = {parsed}\n")
        break

    # ── Rangos ──────────────────────────────
    print("  ─── Rango del parámetro r ───")
    r_min = get_float("r mínimo", -3.0)
    r_max = get_float("r máximo",  3.0)
    if r_min >= r_max:
        r_min, r_max = r_max - 1, r_max

    print("\n  ─── Rango del estado x ───")
    x_min = get_float("x mínimo", -4.0)
    x_max = get_float("x máximo",  4.0)
    if x_min >= x_max:
        x_min, x_max = x_max - 1, x_max

    print("\n  ─── Valor de r para retrato de fase ───")
    r_phase = get_float("r para retrato de fase",
                         round((r_min + r_max) / 2, 3))

    print("\n  ─── Resolución ───")
    n_r    = get_int("Puntos en r (100–2000)", 600)
    n_scan = get_int("Puntos de escaneo en x (200–2000)", 500)
    n_r    = max(50, min(n_r, 2000))
    n_scan = max(100, min(n_scan, 2000))

    r_range = (r_min, r_max)
    x_range = (x_min, x_max)

    # ── Cómputo ─────────────────────────────
    print("\n  ⏳ Calculando equilibrios y estabilidad...")
    stable, unstable, neutral = build_bifurcation_data(
        f, r_range, x_range, n_r=n_r, n_scan=n_scan)

    print("  ⏳ Detectando puntos de bifurcación...")
    bif_pts = detect_bifurcation_points(f, r_range, x_range,
                                         n_r=n_r // 2, n_scan=n_scan // 2)

    # ── Reporte consola ──────────────────────
    print(full_stability_report(f, r_phase, x_range))

    if bif_pts:
        print(f"  ┌─ Puntos de bifurcación detectados ({len(bif_pts)}) ─")
        for rr, eqs in bif_pts[:10]:
            print(f"  │  r ≈ {rr:+.4f}   →   equilibrios: {[round(e, 4) for e in eqs]}")
        if len(bif_pts) > 10:
            print(f"  │  ... y {len(bif_pts)-10} más")
        print("  └" + "─" * 45)
    else:
        print("  ℹ  No se detectaron cambios en el número de equilibrios.")

    # ── Figura ──────────────────────────────
    print("\n  ⏳ Generando gráficos...\n")

    fig = plt.figure(figsize=(16, 10), facecolor=COLORS["bg"])
    fig.suptitle(f"Bifurcación  ·  f(x, r) = {parsed}",
                 fontsize=13, color=COLORS["accent"],
                 fontfamily="monospace", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.38,
                           left=0.07, right=0.97,
                           top=0.92, bottom=0.08)

    ax_bif   = fig.add_subplot(gs[0, :2])   # Diagrama principal (ancho)
    ax_phase = fig.add_subplot(gs[0, 2])    # Retrato de fase
    ax_lam   = fig.add_subplot(gs[1, :2])   # Mapa autovalores
    ax_cnt   = fig.add_subplot(gs[1, 2])    # Conteo de equilibrios

    plot_bifurcation(ax_bif, stable, unstable, neutral, bif_pts, r_range)

    # Línea vertical en r de fase
    ax_bif.axvline(r_phase, color=COLORS["bifurc"], lw=1.5,
                   ls=":", alpha=0.8, label=f"r = {r_phase:.3f} (fase)")
    ax_bif.legend(loc="upper right", fontsize=8,
                  facecolor=COLORS["bg"], edgecolor=COLORS["grid"])

    plot_phase_portrait(ax_phase, f, r_phase, x_range)
    plot_eigenvalue_map(ax_lam, f, r_range, x_range, n_r=n_r)
    plot_number_of_eq(ax_cnt, f, r_range, x_range, n_r=n_r)

    plt.savefig("bifurcacion_resultado.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    print("  💾 Gráfico guardado en: bifurcacion_resultado.png")
    plt.show()

    # ── Exploración adicional ────────────────
    while True:
        ans = input("\n  ¿Analizar otro valor de r? (s/n): ").strip().lower()
        if ans not in ("s", "si", "sí", "y", "yes"):
            break
        r2 = get_float("Nuevo valor de r", r_phase)
        print(full_stability_report(f, r2, x_range))

        fig2, ax2 = plt.subplots(figsize=(7, 4), facecolor=COLORS["bg"])
        plot_phase_portrait(ax2, f, r2, x_range)
        fig2.suptitle(f"f(x, r) = {parsed}", color=COLORS["accent"],
                      fontfamily="monospace")
        plt.tight_layout()
        plt.show()

    print("\n  ✅ Sesión terminada. ¡Hasta la próxima!\n")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()