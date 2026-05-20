"""
Sistema Dinámico 2D No Homogéneo
=================================
Interfaz gráfica para analizar sistemas de la forma:
    x' = a*x + b*y + f(t)
    y' = c*x + d*y + g(t)

Muestra: autovalores, autovectores y diagrama de fase.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import sympy as sp
from sympy import symbols, Matrix, latex, simplify
from sympy import sin, cos, exp, tan, sqrt, pi, E, log, sinh, cosh
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Paleta y estilos
# ─────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
BORDER   = "#30363d"
ACCENT   = "#58a6ff"
ACCENT2  = "#3fb950"
ACCENT3  = "#f78166"
TEXT     = "#e6edf3"
TEXT_DIM = "#8b949e"
ENTRY_BG = "#21262d"
BTN_BG   = "#238636"
BTN_HOV  = "#2ea043"
FONT     = ("Consolas", 11)
FONT_SM  = ("Consolas", 9)
FONT_LG  = ("Consolas", 13, "bold")
FONT_TIT = ("Consolas", 15, "bold")


def make_entry(parent, width=12, **kw):
    e = tk.Entry(parent, width=width, bg=ENTRY_BG, fg=TEXT,
                 insertbackground=ACCENT, relief="flat",
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, font=FONT, **kw)
    return e


def make_label(parent, text, fg=TEXT, font=None, **kw):
    return tk.Label(parent, text=text, bg=PANEL, fg=fg,
                    font=font or FONT, **kw)


def make_section(parent, title):
    frame = tk.LabelFrame(parent, text=f"  {title}  ", bg=PANEL, fg=ACCENT,
                          font=FONT_LG, bd=1, relief="flat",
                          highlightthickness=1, highlightbackground=BORDER,
                          padx=10, pady=8)
    return frame


# ─────────────────────────────────────────────
#  Parseo seguro de expresiones
# ─────────────────────────────────────────────
_SAFE_NAMES = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sqrt": sp.sqrt, "pi": sp.pi, "e": sp.E,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "t": sp.Symbol("t"), "x": sp.Symbol("x"), "y": sp.Symbol("y"),
    "abs": sp.Abs,
}

def parse_expr(text: str, allow_t=True):
    """Parsea una expresión matemática con sympy."""
    text = text.strip().replace("^", "**")
    if not text:
        return sp.Integer(0)
    ns = dict(_SAFE_NAMES)
    if not allow_t:
        ns.pop("t", None)
    try:
        return sp.sympify(text, locals=ns)
    except Exception as e:
        raise ValueError(f"Expresión inválida: '{text}'\n{e}")


# ─────────────────────────────────────────────
#  Motor de análisis
# ─────────────────────────────────────────────
def analyze_system(a_str, b_str, c_str, d_str, f_str, g_str):
    """
    Sistema:  x' = a*x + b*y + f(t)
              y' = c*x + d*y + g(t)
    Retorna dict con toda la información.
    """
    t = sp.Symbol("t")

    # Matriz A
    a = parse_expr(a_str, allow_t=False)
    b = parse_expr(b_str, allow_t=False)
    c = parse_expr(c_str, allow_t=False)
    d = parse_expr(d_str, allow_t=False)

    A = Matrix([[a, b], [c, d]])

    # Términos no homogéneos
    f = parse_expr(f_str, allow_t=True)
    g = parse_expr(g_str, allow_t=True)

    # Autovalores
    char_poly = A.charpoly(sp.Symbol("λ"))
    eig_raw   = A.eigenvals()          # {val: multiplicidad}
    eig_vects = A.eigenvects()         # [(val, mult, [vecs])]

    # Traza y determinante
    tr  = simplify(A.trace())
    det = simplify(A.det())

    # Clasificar el punto de equilibrio (para coefs constantes)
    try:
        tr_n  = complex(tr)
        det_n = complex(det)
        disc  = complex(tr_n**2 - 4*det_n)
        kind  = classify_equilibrium(tr_n.real, det_n.real, disc.real)
    except Exception:
        kind = "Indeterminado (coeficientes simbólicos)"

    # ¿Es homogéneo?
    is_homogeneous = (f == 0 and g == 0)

    return {
        "A": A, "f": f, "g": g,
        "eigenvalues": eig_raw,
        "eigenvects":  eig_vects,
        "trace": tr, "det": det,
        "kind": kind,
        "homogeneous": is_homogeneous,
        "a": a, "b": b, "c": c, "d": d,
    }


def classify_equilibrium(tr, det, disc):
    if det < 0:
        return "Punto de silla (inestable)"
    elif det == 0:
        return "No aislado"
    elif disc > 0:
        if tr > 0:
            return "Nodo inestable (real distinto)"
        elif tr < 0:
            return "Nodo estable (real distinto)"
        else:
            return "Nodo no definido"
    elif disc < 0:
        if tr == 0:
            return "Centro (estable, no asintótico)"
        elif tr > 0:
            return "Espiral inestable"
        else:
            return "Espiral estable"
    else:  # disc == 0
        if tr > 0:
            return "Nodo impropio inestable (raíz repetida)"
        elif tr < 0:
            return "Nodo impropio estable (raíz repetida)"
        else:
            return "Nodo estrella"


# ─────────────────────────────────────────────
#  Graficador de fase
# ─────────────────────────────────────────────
def plot_phase(ax, result, t_max=20, n_traj=12, grid=30):
    a = float(result["a"])
    b = float(result["b"])
    c = float(result["c"])
    d = float(result["d"])

    t  = sp.Symbol("t")
    f_sym = result["f"]
    g_sym = result["g"]

    # Convertir a funciones numéricas
    f_num = sp.lambdify(t, f_sym, "numpy") if f_sym != 0 else lambda t: 0
    g_num = sp.lambdify(t, g_sym, "numpy") if g_sym != 0 else lambda t: 0

    # Campo vectorial (en t=0 para mostrar la estructura)
    lim = 3.0
    xs  = np.linspace(-lim, lim, grid)
    ys  = np.linspace(-lim, lim, grid)
    X, Y = np.meshgrid(xs, ys)
    t0   = 0.0
    U = a*X + b*Y + f_num(t0)
    V = c*X + d*Y + g_num(t0)
    speed = np.sqrt(U**2 + V**2)
    speed[speed == 0] = 1e-10

    ax.streamplot(X, Y, U, V, color=speed, cmap="cool",
                  linewidth=0.8, density=1.2, arrowsize=1.2)

    # Trayectorias desde condiciones iniciales
    from scipy.integrate import solve_ivp
    angles = np.linspace(0, 2*np.pi, n_traj, endpoint=False)
    r0     = 1.5

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, n_traj))

    def system(t_val, y_vec):
        x_, y_ = y_vec
        return [a*x_ + b*y_ + f_num(t_val),
                c*x_ + d*y_ + g_num(t_val)]

    for i, ang in enumerate(angles):
        x0 = r0 * np.cos(ang)
        y0 = r0 * np.sin(ang)
        sol = solve_ivp(system, [0, t_max], [x0, y0],
                        max_step=0.05, dense_output=True)
        if sol.success:
            ts  = np.linspace(0, t_max, 600)
            xy  = sol.sol(ts)
            ax.plot(xy[0], xy[1], color=colors[i], lw=1.0, alpha=0.85)
            ax.plot(xy[0, 0], xy[1, 0], "o", color=colors[i],
                    ms=4, zorder=5)

    # Autovectores
    try:
        A_num = np.array([[a, b], [c, d]], dtype=float)
        vals, vecs = np.linalg.eig(A_num)
        for i, (val, vec) in enumerate(zip(vals, vecs.T)):
            if np.isreal(val):
                v = vec.real / (np.linalg.norm(vec.real) + 1e-12)
                ax.annotate("", xy=(v[0]*lim*0.9, v[1]*lim*0.9),
                            xytext=(-v[0]*lim*0.9, -v[1]*lim*0.9),
                            arrowprops=dict(arrowstyle="-",
                                            color=ACCENT3, lw=1.5,
                                            linestyle="dashed", alpha=0.7))
    except Exception:
        pass

    ax.axhline(0, color=BORDER, lw=0.8, alpha=0.5)
    ax.axvline(0, color=BORDER, lw=0.8, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x", color=TEXT_DIM, fontsize=11)
    ax.set_ylabel("y", color=TEXT_DIM, fontsize=11)
    ax.set_title("Diagrama de Fase", color=TEXT, fontsize=13, pad=10)
    ax.tick_params(colors=TEXT_DIM)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)


# ─────────────────────────────────────────────
#  Ventana principal
# ─────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sistema Dinámico 2D — Análisis No Homogéneo")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("1280x820")
        self._build_ui()

    # ── Layout ──────────────────────────────
    def _build_ui(self):
        # Título
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=20, pady=(14, 0))
        tk.Label(hdr, text="⟨ Sistema Dinámico 2D No Homogéneo ⟩",
                 bg=BG, fg=ACCENT, font=FONT_TIT).pack(side="left")
        tk.Label(hdr, text="x' = ax + by + f(t)   |   y' = cx + dy + g(t)",
                 bg=BG, fg=TEXT_DIM, font=FONT_SM).pack(side="right")

        # Separador
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20, pady=8)

        # Contenedor principal
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=(0, 14))

        # Columna izquierda (controles + resultados)
        left = tk.Frame(main, bg=BG, width=420)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        # Columna derecha (gráfica)
        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_inputs(left)
        self._build_results(left)
        self._build_plot(right)

    def _build_inputs(self, parent):
        sec = make_section(parent, "Matriz del Sistema  A")
        sec.pack(fill="x", pady=(0, 8))

        # Etiqueta sistema
        info = tk.Frame(sec, bg=PANEL)
        info.pack(fill="x", pady=(0, 8))
        tk.Label(info, text="⎡ a  b ⎤ ⎡ x ⎤   ⎡ f(t) ⎤",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_SM).pack(anchor="w")
        tk.Label(info, text="⎣ c  d ⎦ ⎣ y ⎦ + ⎣ g(t) ⎦",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_SM).pack(anchor="w")

        # Grid de entradas
        grid = tk.Frame(sec, bg=PANEL)
        grid.pack(fill="x")

        labels = ["a", "b", "c", "d"]
        self._entries = {}
        defaults = {"a": "-1", "b": "1", "c": "-2", "d": "-1",
                    "f": "0", "g": "0",
                    "t_max": "15", "n_traj": "12", "grid": "28"}

        for i, lbl in enumerate(labels):
            r, col = divmod(i, 2)
            tk.Label(grid, text=f" {lbl} =", bg=PANEL, fg=ACCENT,
                     font=FONT).grid(row=r, column=col*2, sticky="e", pady=3)
            e = make_entry(grid, width=10)
            e.insert(0, defaults[lbl])
            e.grid(row=r, column=col*2+1, padx=(2, 12), pady=3)
            self._entries[lbl] = e

        # No homogéneo
        sec2 = make_section(parent, "Términos No Homogéneos")
        sec2.pack(fill="x", pady=(0, 8))

        for key, label, hint in [("f", "f(t) =", "ej: cos(t), exp(-t)"),
                                   ("g", "g(t) =", "ej: sin(2*t), t")]:
            row = tk.Frame(sec2, bg=PANEL)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, bg=PANEL, fg=ACCENT,
                     font=FONT, width=7, anchor="e").pack(side="left")
            e = make_entry(row, width=22)
            e.insert(0, defaults[key])
            e.pack(side="left", padx=4)
            tk.Label(row, text=hint, bg=PANEL, fg=TEXT_DIM,
                     font=FONT_SM).pack(side="left")
            self._entries[key] = e

        # Opciones de gráfica
        sec3 = make_section(parent, "Opciones de Gráfica")
        sec3.pack(fill="x", pady=(0, 8))

        opts = [("t_max", "Tiempo máx:", "15"),
                ("n_traj", "Nº trayectorias:", "12"),
                ("grid",   "Densidad campo:", "28")]
        for key, lbl, dft in opts:
            row = tk.Frame(sec3, bg=PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=lbl, bg=PANEL, fg=TEXT,
                     font=FONT, width=18, anchor="w").pack(side="left")
            e = make_entry(row, width=8)
            e.insert(0, dft)
            e.pack(side="left")
            self._entries[key] = e

        # Botón Analizar
        btn_frame = tk.Frame(parent, bg=BG)
        btn_frame.pack(fill="x", pady=6)

        btn = tk.Button(btn_frame, text="▶  ANALIZAR SISTEMA",
                        bg=BTN_BG, fg="white", font=FONT_LG,
                        relief="flat", cursor="hand2", bd=0,
                        padx=14, pady=8,
                        activebackground=BTN_HOV, activeforeground="white",
                        command=self.run_analysis)
        btn.pack(fill="x")

        btn_clr = tk.Button(btn_frame, text="↺  Restablecer",
                            bg=ENTRY_BG, fg=TEXT_DIM, font=FONT_SM,
                            relief="flat", cursor="hand2",
                            padx=8, pady=4,
                            activebackground=BORDER,
                            command=self.reset_defaults)
        btn_clr.pack(fill="x", pady=(4, 0))

    def _build_results(self, parent):
        sec = make_section(parent, "Resultados")
        sec.pack(fill="both", expand=True)

        self._result_text = tk.Text(sec, bg=BG, fg=TEXT, font=FONT_SM,
                                    relief="flat", bd=0, wrap="word",
                                    state="disabled",
                                    highlightthickness=0,
                                    insertbackground=ACCENT)
        sb = tk.Scrollbar(sec, command=self._result_text.yview,
                          bg=BORDER, troughcolor=BG)
        self._result_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._result_text.pack(fill="both", expand=True)

        # Tags de colores
        self._result_text.tag_configure("header",
            foreground=ACCENT, font=("Consolas", 11, "bold"))
        self._result_text.tag_configure("value",
            foreground=ACCENT2, font=("Consolas", 10))
        self._result_text.tag_configure("warn",
            foreground=ACCENT3, font=("Consolas", 10))
        self._result_text.tag_configure("dim",
            foreground=TEXT_DIM, font=("Consolas", 9))

    def _build_plot(self, parent):
        self._fig = Figure(figsize=(7, 6), dpi=96,
                           facecolor=PANEL)
        self._ax  = self._fig.add_subplot(111, facecolor=BG)
        self._ax.set_title("Diagrama de Fase", color=TEXT, fontsize=13)
        self._ax.tick_params(colors=TEXT_DIM)
        for sp_ in self._ax.spines.values():
            sp_.set_edgecolor(BORDER)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(parent, bg=PANEL)
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        toolbar.config(background=PANEL)
        toolbar._message_label.config(bg=PANEL, fg=TEXT_DIM)
        toolbar.update()

    # ── Lógica ──────────────────────────────
    def get(self, key):
        return self._entries[key].get().strip()

    def run_analysis(self):
        try:
            result = analyze_system(
                self.get("a"), self.get("b"),
                self.get("c"), self.get("d"),
                self.get("f"), self.get("g"),
            )
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e))
            return

        self._show_results(result)
        self._update_plot(result)

    def _show_results(self, res):
        txt = self._result_text
        txt.configure(state="normal")
        txt.delete("1.0", "end")

        def line(text, tag=""):
            txt.insert("end", text + "\n", tag)

        line("══════════════════════════════", "dim")
        line(" MATRIZ DEL SISTEMA", "header")
        line("══════════════════════════════", "dim")
        A = res["A"]
        line(f"  A = ⎡ {A[0,0]}  {A[0,1]} ⎤", "value")
        line(f"      ⎣ {A[1,0]}  {A[1,1]} ⎦", "value")

        line("")
        line("  Traza   = " + str(res["trace"]), "value")
        line("  Det(A)  = " + str(res["det"]),   "value")

        line("")
        line("══════════════════════════════", "dim")
        line(" AUTOVALORES  (λ)", "header")
        line("══════════════════════════════", "dim")
        for val, mult in res["eigenvalues"].items():
            val_s = sp.nsimplify(val, rational=False)
            line(f"  λ = {val_s}   (mult. {mult})", "value")

        line("")
        line("══════════════════════════════", "dim")
        line(" AUTOVECTORES", "header")
        line("══════════════════════════════", "dim")
        for val, mult, vecs in res["eigenvects"]:
            val_s = sp.nsimplify(val, rational=False)
            line(f"  λ = {val_s}  →", "value")
            for v in vecs:
                v_norm = simplify(v)
                line(f"    v = {list(v_norm)}", "value")

        line("")
        line("══════════════════════════════", "dim")
        line(" CLASIFICACIÓN  (equilibrio)", "header")
        line("══════════════════════════════", "dim")
        line(f"  {res['kind']}", "warn")

        line("")
        line("══════════════════════════════", "dim")
        line(" TÉRMINOS NO HOMOGÉNEOS", "header")
        line("══════════════════════════════", "dim")
        if res["homogeneous"]:
            line("  Sistema HOMOGÉNEO  (f=g=0)", "dim")
        else:
            line(f"  f(t) = {res['f']}", "value")
            line(f"  g(t) = {res['g']}", "value")

        line("")
        line("══════════════════════════════", "dim")

        txt.configure(state="disabled")

    def _update_plot(self, result):
        self._ax.clear()
        try:
            t_max  = float(self.get("t_max"))
            n_traj = int(self.get("n_traj"))
            grid   = int(self.get("grid"))

            # Verificar que los coeficientes sean numéricos
            float(result["a"])
            float(result["b"])
            float(result["c"])
            float(result["d"])

            plot_phase(self._ax, result, t_max=t_max,
                       n_traj=n_traj, grid=grid)
        except Exception as e:
            self._ax.text(0.5, 0.5,
                f"No se puede graficar:\n{e}",
                ha="center", va="center",
                color=ACCENT3, fontsize=10,
                transform=self._ax.transAxes)
            self._ax.set_facecolor(BG)

        self._fig.tight_layout()
        self._canvas.draw()

    def reset_defaults(self):
        defaults = {"a": "-1", "b": "1", "c": "-2", "d": "-1",
                    "f": "0", "g": "0",
                    "t_max": "15", "n_traj": "12", "grid": "28"}
        for k, v in defaults.items():
            self._entries[k].delete(0, "end")
            self._entries[k].insert(0, v)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()