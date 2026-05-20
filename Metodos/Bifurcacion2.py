"""
╔══════════════════════════════════════════════════════════════╗
║           ANALIZADOR DE BIFURCACIONES DINÁMICAS              ║
║     Diagrama de bifurcación · Diagrama de fase · Estabilidad ║
╚══════════════════════════════════════════════════════════════╝

Uso:
  python bifurcacion_analyzer.py

Dependencias:
  pip install numpy matplotlib scipy sympy
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import brentq, fsolve
from scipy.integrate import odeint
import sympy as sp
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PALETA DE COLORES
# ─────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
ACCENT   = "#58a6ff"
ACCENT2  = "#f78166"
ACCENT3  = "#3fb950"
MUTED    = "#8b949e"
TEXT     = "#e6edf3"
BORDER   = "#30363d"
STABLE   = "#3fb950"
UNSTABLE = "#f85149"
SADDLE   = "#d29922"

# ─────────────────────────────────────────────
#  EJEMPLOS PREDEFINIDOS
# ─────────────────────────────────────────────
EJEMPLOS = {
    "Bifurcación de Silla-Nodo": {
        "formula": "r + x**2",
        "r_min": "-2", "r_max": "2",
        "x_min": "-3", "x_max": "3",
        "descripcion": (
            "dx/dt = r + x²\n"
            "Punto fijo: x* = ±√(-r)\n"
            "• r < 0 → dos puntos fijos (estable e inestable)\n"
            "• r = 0 → bifurcación (un punto fijo semi-estable)\n"
            "• r > 0 → sin puntos fijos reales"
        ),
    },
    "Bifurcación de Transcrítica": {
        "formula": "r*x - x**2",
        "r_min": "-2", "r_max": "2",
        "x_min": "-3", "x_max": "3",
        "descripcion": (
            "dx/dt = rx - x²\n"
            "Puntos fijos: x* = 0 y x* = r\n"
            "• r < 0 → x*=0 estable, x*=r inestable\n"
            "• r = 0 → bifurcación (intercambio de estabilidad)\n"
            "• r > 0 → x*=0 inestable, x*=r estable"
        ),
    },
    "Bifurcación de Pitchfork (Supercrítica)": {
        "formula": "r*x - x**3",
        "r_min": "-2", "r_max": "2",
        "x_min": "-2", "x_max": "2",
        "descripcion": (
            "dx/dt = rx - x³  (supercrítica)\n"
            "Puntos fijos: x*=0 y x*=±√r (r>0)\n"
            "• r < 0 → solo x*=0 estable\n"
            "• r = 0 → bifurcación pitchfork\n"
            "• r > 0 → x*=0 inestable, ±√r estables"
        ),
    },
    "Bifurcación de Pitchfork (Subcrítica)": {
        "formula": "r*x + x**3",
        "r_min": "-2", "r_max": "2",
        "x_min": "-2", "x_max": "2",
        "descripcion": (
            "dx/dt = rx + x³  (subcrítica)\n"
            "Puntos fijos: x*=0 y x*=±√(-r) (r<0)\n"
            "• r < 0 → x*=0 estable, ±√(-r) inestables\n"
            "• r = 0 → bifurcación subcrítica\n"
            "• r > 0 → x*=0 inestable (explosión)"
        ),
    },
    "Oscilador de Van der Pol": {
        "formula": "r*x - x**3/3 - x**5",
        "r_min": "-1", "r_max": "3",
        "x_min": "-3", "x_max": "3",
        "descripcion": (
            "dx/dt = rx - x³/3 - x⁵\n"
            "Dinámica compleja con múltiples puntos fijos\n"
            "y posible histéresis (bifurcación subcrítica)"
        ),
    },
    "Logística Discreta (Caos)": {
        "formula": "r*x*(1 - x)",
        "r_min": "0", "r_max": "4",
        "x_min": "0", "x_max": "1",
        "descripcion": (
            "Mapa logístico: xₙ₊₁ = r·xₙ·(1−xₙ)\n"
            "• 0 < r < 1  → extinción\n"
            "• 1 < r < 3  → punto fijo estable\n"
            "• 3 < r < 3.57 → duplicación de período\n"
            "• r > 3.57  → caos"
        ),
    },
    "Personalizado": {
        "formula": "",
        "r_min": "-2", "r_max": "2",
        "x_min": "-3", "x_max": "3",
        "descripcion": "Ingresa tu propia fórmula f(x, r).\nUsa 'x' como variable de estado y 'r' como parámetro.",
    },
}


# ═══════════════════════════════════════════════════════════
#   MOTOR DE ANÁLISIS
# ═══════════════════════════════════════════════════════════
class BifurcationEngine:
    def __init__(self, formula_str: str, r_range, x_range):
        self.formula_str = formula_str.strip()
        self.r_range = r_range
        self.x_range = x_range
        self._build_functions()

    def _build_functions(self):
        x, r = sp.symbols("x r")
        try:
            expr = sp.sympify(self.formula_str, locals={"x": x, "r": r})
        except Exception as e:
            raise ValueError(f"Fórmula inválida: {e}")

        self.expr     = expr
        self.expr_dx  = sp.diff(expr, x)
        self.f        = sp.lambdify((x, r), expr, "numpy")
        self.df       = sp.lambdify((x, r), self.expr_dx, "numpy")

    # ── Puntos fijos para un r dado ──────────────────────────
    def fixed_points(self, r_val, n_seeds=400):
        xmin, xmax = self.x_range
        seeds = np.linspace(xmin, xmax, n_seeds)
        fps   = []
        try:
            fvals = self.f(seeds, r_val)
        except Exception:
            return fps

        # Búsqueda de cambios de signo → Brent
        for i in range(len(seeds) - 1):
            if np.isfinite(fvals[i]) and np.isfinite(fvals[i+1]):
                if fvals[i] * fvals[i+1] < 0:
                    try:
                        root = brentq(lambda xx: float(self.f(xx, r_val)),
                                      seeds[i], seeds[i+1], xtol=1e-10)
                        fps.append(root)
                    except Exception:
                        pass

        # fsolve adicional para puntos no detectados
        for s in np.linspace(xmin, xmax, 20):
            try:
                sol = fsolve(lambda xx: float(self.f(xx[0], r_val)),
                             [s], full_output=True)
                if sol[2] == 1:
                    root = sol[0][0]
                    if xmin <= root <= xmax:
                        if not any(abs(root - p) < 1e-6 for p in fps):
                            fps.append(root)
            except Exception:
                pass

        return sorted(fps)

    # ── Estabilidad de un punto fijo ─────────────────────────
    def stability(self, x_val, r_val):
        try:
            deriv = float(self.df(x_val, r_val))
            if   deriv < -1e-6: return "stable",   deriv
            elif deriv >  1e-6: return "unstable", deriv
            else:               return "semistable", deriv
        except Exception:
            return "unknown", np.nan

    # ── Diagrama de bifurcación completo ─────────────────────
    def bifurcation_diagram(self, n_r=600, n_seeds=300):
        r_vals   = np.linspace(*self.r_range, n_r)
        stable   = {"r": [], "x": []}
        unstable = {"r": [], "x": []}
        semi     = {"r": [], "x": []}
        critical = []

        prev_fps = None
        for r_val in r_vals:
            fps = self.fixed_points(r_val, n_seeds)
            # Detectar cambio en número de puntos fijos → bifurcación
            if prev_fps is not None and len(fps) != len(prev_fps):
                critical.append(r_val)
            prev_fps = fps

            for xp in fps:
                st, _ = self.stability(xp, r_val)
                if   st == "stable":     stable["r"].append(r_val);   stable["x"].append(xp)
                elif st == "unstable":   unstable["r"].append(r_val); unstable["x"].append(xp)
                else:                    semi["r"].append(r_val);      semi["x"].append(xp)

        return stable, unstable, semi, critical

    # ── Campo vectorial / diagrama de fase 1D ────────────────
    def phase_portrait_1d(self, r_val, n=500):
        x_arr = np.linspace(*self.x_range, n)
        try:
            dx = self.f(x_arr, r_val)
        except Exception:
            dx = np.zeros_like(x_arr)
        return x_arr, dx

    # ── Órbita temporal ──────────────────────────────────────
    def time_series(self, x0, r_val, t_end=30, n=3000):
        t = np.linspace(0, t_end, n)
        try:
            sol = odeint(lambda xx, tt: float(self.f(xx[0], r_val)), [x0], t,
                         rtol=1e-8, atol=1e-10)
            return t, sol[:, 0]
        except Exception:
            return t, np.full_like(t, np.nan)

    # ── Mapa logístico (modo discreto) ──────────────────────
    def logistic_map(self, n_r=800, n_iter=500, n_discard=200):
        r_vals  = np.linspace(*self.r_range, n_r)
        x0      = 0.5
        r_plot  = []
        x_plot  = []
        for r_val in r_vals:
            x = x0
            for _ in range(n_discard):
                try:  x = float(self.f(x, r_val))
                except: break
                if not np.isfinite(x): break
            for _ in range(n_iter):
                try:  x = float(self.f(x, r_val))
                except: break
                if not np.isfinite(x): break
                r_plot.append(r_val)
                x_plot.append(x)
        return np.array(r_plot), np.array(x_plot)


# ═══════════════════════════════════════════════════════════
#   INTERFAZ GRÁFICA
# ═══════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analizador de Bifurcaciones")
        self.configure(bg=BG)
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self._engine   = None
        self._is_map   = False
        self._build_ui()
        self._load_example("Bifurcación de Silla-Nodo")

    # ────────────────────────────────────────────────────────
    #  CONSTRUCCIÓN DE UI
    # ────────────────────────────────────────────────────────
    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Panel izquierdo (controles) ──────────────────────
        left = tk.Frame(self, bg=PANEL, width=310)
        left.grid(row=0, column=0, sticky="ns", padx=0, pady=0)
        left.grid_propagate(False)
        left.rowconfigure(99, weight=1)

        # Título
        tk.Label(left, text="⚡ BIFURCACIONES", font=("Courier", 14, "bold"),
                 bg=PANEL, fg=ACCENT).pack(pady=(18,2))
        tk.Label(left, text="Análisis de Sistemas Dinámicos", font=("Courier", 8),
                 bg=PANEL, fg=MUTED).pack(pady=(0,14))

        self._sep(left)

        # ── Ejemplos ─────────────────────────────────────────
        tk.Label(left, text="EJEMPLOS", font=("Courier", 9, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14, pady=(10,4))

        self.ejemplo_var = tk.StringVar(value="Bifurcación de Silla-Nodo")
        combo_style = {"bg": "#1c2128", "fg": TEXT, "font": ("Courier", 9),
                       "relief": "flat", "bd": 0}
        self.combo = ttk.Combobox(left, textvariable=self.ejemplo_var,
                                  values=list(EJEMPLOS.keys()), state="readonly",
                                  font=("Courier", 9), width=32)
        self.combo.pack(padx=14, pady=(0,6))
        self.combo.bind("<<ComboboxSelected>>", self._on_ejemplo)

        self._sep(left)

        # ── Fórmula ──────────────────────────────────────────
        tk.Label(left, text="FÓRMULA  f(x, r)", font=("Courier", 9, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14, pady=(10,2))
        tk.Label(left, text="dx/dt = f(x, r)   →  usa 'x' y 'r'",
                 font=("Courier", 8), bg=PANEL, fg=MUTED).pack(anchor="w", padx=14)

        self.formula_var = tk.StringVar()
        entry_f = tk.Entry(left, textvariable=self.formula_var,
                           bg="#1c2128", fg=ACCENT, font=("Courier", 11),
                           insertbackground=ACCENT, relief="flat",
                           highlightthickness=1, highlightbackground=BORDER,
                           highlightcolor=ACCENT)
        entry_f.pack(fill="x", padx=14, pady=(4,2))

        self._sep(left)

        # ── Rangos ───────────────────────────────────────────
        tk.Label(left, text="RANGOS", font=("Courier", 9, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14, pady=(10,4))

        self.r_min_var = tk.StringVar(value="-2")
        self.r_max_var = tk.StringVar(value="2")
        self.x_min_var = tk.StringVar(value="-3")
        self.x_max_var = tk.StringVar(value="3")

        for label, vmin, vmax in [
            ("r  (parámetro)", self.r_min_var, self.r_max_var),
            ("x  (estado)",    self.x_min_var, self.x_max_var),
        ]:
            row = tk.Frame(left, bg=PANEL)
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=label, font=("Courier", 8), bg=PANEL, fg=TEXT,
                     width=16, anchor="w").pack(side="left")
            for var, hint in [(vmin, "min"), (vmax, "max")]:
                tk.Label(row, text=hint, font=("Courier", 7), bg=PANEL,
                         fg=MUTED).pack(side="left", padx=(4,1))
                tk.Entry(row, textvariable=var, width=6,
                         bg="#1c2128", fg=ACCENT2, font=("Courier", 9),
                         insertbackground=ACCENT2, relief="flat").pack(side="left")

        self._sep(left)

        # ── Modo ──────────────────────────────────────────────
        tk.Label(left, text="MODO DE ANÁLISIS", font=("Courier", 9, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14, pady=(10,4))

        self.mode_var = tk.StringVar(value="continuo")
        modes = [("Sistema continuo  dx/dt = f", "continuo"),
                 ("Mapa discreto  xₙ₊₁ = f(xₙ, r)", "discreto")]
        for txt, val in modes:
            tk.Radiobutton(left, text=txt, variable=self.mode_var, value=val,
                           bg=PANEL, fg=TEXT, selectcolor=PANEL,
                           activebackground=PANEL, activeforeground=ACCENT,
                           font=("Courier", 8)).pack(anchor="w", padx=20)

        self._sep(left)

        # ── r para diagrama de fase ───────────────────────────
        tk.Label(left, text="r  PARA DIAGRAMA DE FASE", font=("Courier", 9, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14, pady=(10,2))

        slider_fr = tk.Frame(left, bg=PANEL)
        slider_fr.pack(fill="x", padx=14)
        self.r_phase_var = tk.DoubleVar(value=0.5)
        self.r_phase_lbl = tk.Label(slider_fr, text="r = 0.50",
                                    font=("Courier", 10, "bold"),
                                    bg=PANEL, fg=ACCENT)
        self.r_phase_lbl.pack(side="right")
        self.r_slider = tk.Scale(slider_fr, from_=-2, to=2,
                                  resolution=0.01, orient="horizontal",
                                  variable=self.r_phase_var,
                                  bg=PANEL, fg=TEXT, troughcolor="#1c2128",
                                  highlightthickness=0, bd=0, showvalue=False,
                                  command=self._on_slider)
        self.r_slider.pack(side="left", fill="x", expand=True)

        # ── x₀ para serie temporal ────────────────────────────
        tk.Label(left, text="x₀  CONDICIÓN INICIAL", font=("Courier", 9, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14, pady=(8,2))
        self.x0_var = tk.StringVar(value="0.5")
        tk.Entry(left, textvariable=self.x0_var, bg="#1c2128", fg=ACCENT3,
                 font=("Courier", 10), insertbackground=ACCENT3,
                 relief="flat").pack(fill="x", padx=14)

        self._sep(left)

        # ── Botones ───────────────────────────────────────────
        btn_cfg = {"font": ("Courier", 10, "bold"), "relief": "flat",
                   "cursor": "hand2", "pady": 8}

        tk.Button(left, text="▶  ANALIZAR", bg=ACCENT, fg=BG,
                  command=self._run, **btn_cfg).pack(fill="x", padx=14, pady=(6,2))

        tk.Button(left, text="⟳  LIMPIAR", bg=BORDER, fg=TEXT,
                  command=self._clear, **btn_cfg).pack(fill="x", padx=14, pady=2)

        tk.Button(left, text="💾  GUARDAR GRÁFICO", bg=BORDER, fg=ACCENT2,
                  command=self._save, **btn_cfg).pack(fill="x", padx=14, pady=2)

        self._sep(left)

        # ── Log / Descripción ─────────────────────────────────
        tk.Label(left, text="ANÁLISIS", font=("Courier", 9, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=14, pady=(6,2))
        self.log = scrolledtext.ScrolledText(left, bg="#0d1117", fg=TEXT,
                                             font=("Courier", 8),
                                             relief="flat", height=12,
                                             insertbackground=TEXT)
        self.log.pack(fill="both", expand=True, padx=8, pady=(0,8))

        # ── Panel derecho (gráficos) ──────────────────────────
        right = tk.Frame(self, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(10, 8), facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(right, bg=BG)
        toolbar_frame.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

    # ────────────────────────────────────────────────────────
    #  HELPERS
    # ────────────────────────────────────────────────────────
    def _sep(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=8, pady=4)

    def _log(self, msg, color=TEXT):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def _clear_log(self):
        self.log.delete("1.0", "end")

    def _on_ejemplo(self, _=None):
        name = self.ejemplo_var.get()
        self._load_example(name)

    def _load_example(self, name):
        ex = EJEMPLOS[name]
        self.formula_var.set(ex["formula"])
        self.r_min_var.set(ex["r_min"])
        self.r_max_var.set(ex["r_max"])
        self.x_min_var.set(ex["x_min"])
        self.x_max_var.set(ex["x_max"])
        self._clear_log()
        self._log(f"═══ {name} ═══\n")
        self._log(ex["descripcion"])
        # Ajustar slider
        r_mn = float(ex["r_min"])
        r_mx = float(ex["r_max"])
        mid  = (r_mn + r_mx) / 2
        self.r_slider.config(from_=r_mn, to=r_mx, resolution=(r_mx-r_mn)/200)
        self.r_phase_var.set(mid)
        self.r_phase_lbl.config(text=f"r = {mid:.2f}")
        # Modo automático para logístico
        if "Logística" in name or "Caos" in name:
            self.mode_var.set("discreto")
        else:
            self.mode_var.set("continuo")

    def _on_slider(self, val):
        rv = float(val)
        self.r_phase_lbl.config(text=f"r = {rv:.2f}")

    # ────────────────────────────────────────────────────────
    #  ANÁLISIS PRINCIPAL
    # ────────────────────────────────────────────────────────
    def _run(self):
        formula = self.formula_var.get().strip()
        if not formula:
            messagebox.showerror("Error", "Ingresa una fórmula f(x, r)")
            return
        try:
            r_range = (float(self.r_min_var.get()), float(self.r_max_var.get()))
            x_range = (float(self.x_min_var.get()), float(self.x_max_var.get()))
            x0      = float(self.x0_var.get())
            r_phase = float(self.r_phase_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Valor inválido: {e}")
            return

        try:
            engine = BifurcationEngine(formula, r_range, x_range)
        except ValueError as e:
            messagebox.showerror("Error en fórmula", str(e))
            return

        self._engine = engine
        self._clear_log()
        self._log(f"▶ Fórmula: f(x,r) = {formula}\n")

        is_discrete = (self.mode_var.get() == "discreto")
        self._is_map = is_discrete

        self.fig.clear()

        if is_discrete:
            self._plot_discrete(engine, r_range, x_range, r_phase)
        else:
            self._plot_continuous(engine, r_range, x_range, r_phase, x0)

        self.canvas.draw()

    # ────────────────────────────────────────────────────────
    #  GRÁFICOS CONTINUOS
    # ────────────────────────────────────────────────────────
    def _plot_continuous(self, engine, r_range, x_range, r_phase, x0):
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.42, wspace=0.35,
                               left=0.08, right=0.97, top=0.93, bottom=0.08)
        ax1 = self.fig.add_subplot(gs[0, :])   # diagrama bifurcación (arriba completo)
        ax2 = self.fig.add_subplot(gs[1, 0])   # diagrama de fase
        ax3 = self.fig.add_subplot(gs[1, 1])   # serie temporal

        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor("#0d1117")
            for sp_ in ax.spines.values():
                sp_.set_color(BORDER)
            ax.tick_params(colors=MUTED, labelsize=8)

        # ── Diagrama de bifurcación ──────────────────────────
        self._log("⏳ Calculando diagrama de bifurcación...")
        self.update()
        stable, unstable, semi, crit = engine.bifurcation_diagram(n_r=700)

        if stable["r"]:
            ax1.scatter(stable["r"], stable["x"], s=2, c=STABLE,
                        label="Estable", zorder=3, alpha=0.85)
        if unstable["r"]:
            ax1.scatter(unstable["r"], unstable["x"], s=2, c=UNSTABLE,
                        label="Inestable", zorder=3, alpha=0.85)
        if semi["r"]:
            ax1.scatter(semi["r"], semi["x"], s=2, c=SADDLE,
                        label="Semi-estable", zorder=3, alpha=0.85)

        for rc in crit:
            ax1.axvline(rc, color=ACCENT, lw=1.2, alpha=0.6, ls="--")

        ax1.axvline(r_phase, color=ACCENT2, lw=1.8, alpha=0.9,
                    label=f"r = {r_phase:.2f}", zorder=5)
        ax1.set_xlabel("r  (parámetro)", color=MUTED, fontsize=9)
        ax1.set_ylabel("x*  (puntos fijos)", color=MUTED, fontsize=9)
        ax1.set_title("DIAGRAMA DE BIFURCACIÓN", color=TEXT,
                      fontsize=11, fontweight="bold", pad=8)
        ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER,
                   labelcolor=TEXT, markerscale=5)
        ax1.set_xlim(r_range)
        ax1.set_ylim(x_range)

        # ── Diagrama de fase 1D ──────────────────────────────
        self._log(f"⏳ Calculando diagrama de fase (r = {r_phase:.2f})...")
        self.update()
        x_arr, dx_arr = engine.phase_portrait_1d(r_phase)
        ax2.plot(x_arr, dx_arr, color=ACCENT, lw=2, zorder=3)
        ax2.axhline(0, color=MUTED, lw=1, ls="--", alpha=0.6)
        ax2.axvline(0, color=MUTED, lw=0.6, alpha=0.4)
        ax2.fill_between(x_arr, 0, dx_arr,
                         where=(dx_arr > 0), alpha=0.12, color=STABLE)
        ax2.fill_between(x_arr, 0, dx_arr,
                         where=(dx_arr < 0), alpha=0.12, color=UNSTABLE)

        fps = engine.fixed_points(r_phase)
        stability_report = []
        for xp in fps:
            st, deriv = engine.stability(xp, r_phase)
            color = STABLE if st == "stable" else (UNSTABLE if st == "unstable" else SADDLE)
            marker = "o" if st == "stable" else ("x" if st == "unstable" else "D")
            ax2.scatter([xp], [0], s=110, c=color, zorder=6,
                        marker=marker, linewidths=2)
            stability_report.append((xp, st, deriv))

        # Flechas de flujo
        n_arrows = 12
        for xa in np.linspace(*x_range, n_arrows):
            dxa = float(engine.f(xa, r_phase)) if np.isfinite(xa) else 0
            sign = np.sign(dxa)
            if sign == 0: continue
            dx_arrow = sign * (x_range[1]-x_range[0]) * 0.05
            ax2.annotate("", xy=(xa + dx_arrow, 0), xytext=(xa, 0),
                         arrowprops=dict(arrowstyle="->", color=ACCENT2,
                                         lw=1.2, alpha=0.7))

        ax2.set_xlabel("x", color=MUTED, fontsize=9)
        ax2.set_ylabel("f(x, r)", color=MUTED, fontsize=9)
        ax2.set_title(f"DIAGRAMA DE FASE  (r = {r_phase:.2f})",
                      color=TEXT, fontsize=10, fontweight="bold")
        ax2.set_xlim(x_range)

        # ── Serie temporal ───────────────────────────────────
        self._log(f"⏳ Integrando trayectoria (x₀ = {x0})...")
        self.update()
        t, xt = engine.time_series(x0, r_phase)
        ax3.plot(t, xt, color=ACCENT3, lw=1.5)
        ax3.axhline(0, color=MUTED, lw=0.6, ls="--", alpha=0.4)
        for xp, st, _ in stability_report:
            color = STABLE if st == "stable" else UNSTABLE
            ax3.axhline(xp, color=color, lw=1, ls=":", alpha=0.7)
        ax3.set_xlabel("t", color=MUTED, fontsize=9)
        ax3.set_ylabel("x(t)", color=MUTED, fontsize=9)
        ax3.set_title(f"SERIE TEMPORAL  (x₀ = {x0})",
                      color=TEXT, fontsize=10, fontweight="bold")

        self.fig.patch.set_facecolor(BG)

        # ── Log de resultados ─────────────────────────────────
        self._log(f"\n{'─'*40}")
        self._log(f"PUNTOS FIJOS  (r = {r_phase:.4f})")
        self._log(f"{'─'*40}")
        if stability_report:
            for xp, st, deriv in stability_report:
                icon = "●" if st == "stable" else ("✕" if st == "unstable" else "◆")
                self._log(f"  {icon}  x* = {xp:+.6f}")
                self._log(f"       f'(x*) = {deriv:+.6f}")
                self._log(f"       Estado: {st.upper()}")
        else:
            self._log("  (sin puntos fijos en el rango dado)")

        if crit:
            self._log(f"\n{'─'*40}")
            self._log("PUNTOS CRÍTICOS DE BIFURCACIÓN")
            self._log(f"{'─'*40}")
            for rc in crit[:10]:
                self._log(f"  r* ≈ {rc:.6f}")

        self._log("\n✓ Análisis completado.")

    # ────────────────────────────────────────────────────────
    #  GRÁFICOS DISCRETOS (MAPA LOGÍSTICO)
    # ────────────────────────────────────────────────────────
    def _plot_discrete(self, engine, r_range, x_range, r_phase):
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.42, wspace=0.35,
                               left=0.08, right=0.97, top=0.93, bottom=0.08)
        ax1 = self.fig.add_subplot(gs[0, :])
        ax2 = self.fig.add_subplot(gs[1, 0])
        ax3 = self.fig.add_subplot(gs[1, 1])

        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor("#0d1117")
            for sp_ in ax.spines.values():
                sp_.set_color(BORDER)
            ax.tick_params(colors=MUTED, labelsize=8)

        # Mapa de bifurcación
        self._log("⏳ Calculando mapa de bifurcación (discreto)...")
        self.update()
        r_p, x_p = engine.logistic_map(n_r=1000, n_iter=300, n_discard=200)
        ax1.scatter(r_p, x_p, s=0.3, c=ACCENT, alpha=0.5)
        ax1.axvline(r_phase, color=ACCENT2, lw=1.8, alpha=0.9,
                    label=f"r = {r_phase:.2f}")
        ax1.set_xlabel("r  (parámetro)", color=MUTED, fontsize=9)
        ax1.set_ylabel("xₙ (atractor)", color=MUTED, fontsize=9)
        ax1.set_title("DIAGRAMA DE BIFURCACIÓN (MAPA DISCRETO)",
                      color=TEXT, fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        ax1.set_xlim(r_range)
        ax1.set_ylim(x_range)

        # Diagrama de tela de araña (cobweb)
        self._log(f"⏳ Construyendo cobweb (r = {r_phase:.2f})...")
        self.update()
        x_arr = np.linspace(*x_range, 400)
        try:
            y_arr = np.array([float(engine.f(xx, r_phase)) for xx in x_arr])
        except:
            y_arr = np.zeros_like(x_arr)
        ax2.plot(x_arr, y_arr, color=ACCENT, lw=2, label="f(x)", zorder=3)
        ax2.plot(x_arr, x_arr, color=MUTED, lw=1, ls="--", label="y = x", zorder=2)
        # Iteraciones cobweb
        x_cob = 0.5
        n_cob = 60
        cx, cy = [x_cob], [0]
        for _ in range(n_cob):
            yn = float(engine.f(x_cob, r_phase))
            cx += [x_cob, yn]; cy += [yn, yn]
            x_cob = yn
            if not np.isfinite(x_cob) or not (x_range[0] <= x_cob <= x_range[1]):
                break
        ax2.plot(cx, cy, color=ACCENT2, lw=0.9, alpha=0.75, zorder=4)
        ax2.set_xlabel("xₙ", color=MUTED, fontsize=9)
        ax2.set_ylabel("xₙ₊₁", color=MUTED, fontsize=9)
        ax2.set_title(f"DIAGRAMA COBWEB  (r = {r_phase:.2f})",
                      color=TEXT, fontsize=10, fontweight="bold")
        ax2.legend(fontsize=8, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        ax2.set_xlim(x_range); ax2.set_ylim(x_range)

        # Serie temporal discreta
        self._log("⏳ Generando serie temporal discreta...")
        self.update()
        x_ts = [0.5]
        for _ in range(150):
            try:
                xn = float(engine.f(x_ts[-1], r_phase))
            except:
                break
            if not np.isfinite(xn): break
            x_ts.append(xn)
        ax3.plot(range(len(x_ts)), x_ts, color=ACCENT3, lw=1.3, marker="o",
                 markersize=2, alpha=0.85)
        ax3.set_xlabel("n", color=MUTED, fontsize=9)
        ax3.set_ylabel("xₙ", color=MUTED, fontsize=9)
        ax3.set_title(f"SERIE TEMPORAL DISCRETA  (r = {r_phase:.2f})",
                      color=TEXT, fontsize=10, fontweight="bold")

        self.fig.patch.set_facecolor(BG)
        self._log("\n✓ Análisis discreto completado.")
        self._log(f"\n  r actual = {r_phase:.4f}")
        unique_x = set(round(v, 4) for v in x_ts[-50:])
        if len(unique_x) == 1:
            self._log(f"  Atractor: punto fijo x* ≈ {list(unique_x)[0]}")
        elif len(unique_x) <= 8:
            self._log(f"  Ciclo de período {len(unique_x)}")
        else:
            self._log(f"  Comportamiento caótico ({len(unique_x)} valores únicos)")

    # ────────────────────────────────────────────────────────
    #  ACCIONES
    # ────────────────────────────────────────────────────────
    def _clear(self):
        self.fig.clear()
        self.canvas.draw()
        self._clear_log()

    def _save(self):
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            title="Guardar gráfico",
        )
        if path:
            self.fig.savefig(path, dpi=180, facecolor=BG, bbox_inches="tight")
            self._log(f"\n💾 Guardado: {path}")


# ═══════════════════════════════════════════════════════════
#   ENTRADA
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        import sympy, scipy, matplotlib, numpy
    except ImportError as e:
        print(f"\n[ERROR] Dependencia faltante: {e}")
        print("Instala con:  pip install numpy matplotlib scipy sympy\n")
        raise

    app = App()
    app.mainloop()