"""
Sistema Dinámico 2D  —  Afín
=============================
Sistema:  ẋ = Ax + b
          donde A es 2×2  y  b ∈ ℝ²

El punto de equilibrio satisface  Ax* + b = 0  →  x* = -A⁻¹b

Requiere: numpy, matplotlib, scipy
Instalar: pip install numpy matplotlib scipy
"""

import tkinter as tk
from tkinter import font as tkfont
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings("ignore")

# ─── Paleta ───────────────────────────────────────────────────────────────────
BG       = "#0f1117"
PANEL    = "#1a1d27"
ACCENT   = "#4fc3f7"
ACCENT2  = "#81d4fa"
GREEN    = "#69f0ae"
RED      = "#ff5252"
YELLOW   = "#ffeb3b"
ORANGE   = "#ffb74d"
GRAY     = "#546e7a"
TEXT     = "#eceff1"
TEXT_DIM = "#90a4ae"
BORDER   = "#263238"


def _entry(parent, var, width=6):
    return tk.Entry(parent, textvariable=var, width=width,
                    bg="#0d1117", fg=ACCENT, insertbackground=ACCENT,
                    font=("Courier", 11), relief="flat",
                    highlightthickness=1, highlightcolor=ACCENT,
                    highlightbackground=BORDER)


class SistemaDinamico2D:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sistema Dinámico 2D  —  ẋ = Ax + b")
        self.root.configure(bg=BG)
        self.root.geometry("1320x820")
        self.root.minsize(960, 640)

        self.f_title = tkfont.Font(family="Courier", size=12, weight="bold")
        self.f_mono  = tkfont.Font(family="Courier", size=11)
        self.f_small = tkfont.Font(family="Courier", size=9)
        self.f_label = tkfont.Font(family="Courier", size=10)

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_graph()
        self._actualizar()

    # ──────────────────────────────────────────────────────────────────────────
    # Sidebar
    # ──────────────────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = tk.Frame(self.root, bg=PANEL, width=330)
        sb.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        sb.grid_propagate(False)
        sb.columnconfigure(0, weight=1)
        self._sb = sb
        row = 0

        # Título
        tk.Label(sb, text="[ SISTEMA DINÁMICO 2D ]",
                 bg=PANEL, fg=ACCENT, font=self.f_title,
                 pady=8).grid(row=row, column=0, sticky="ew"); row += 1

        self.lbl_ecuacion = tk.Label(sb,
            text="ẋ = ax + by + b₁\nẏ = cx + dy + b₂",
            bg=PANEL, fg=TEXT_DIM, font=self.f_small, justify="center")
        self.lbl_ecuacion.grid(row=row, column=0, sticky="ew", pady=(0, 6)); row += 1

        self._sep(row); row += 1

        # ── Matriz A ──────────────────────────────────────────────────────────
        tk.Label(sb, text="▸ MATRIZ  A",
                 bg=PANEL, fg=ACCENT2, font=self.f_label,
                 anchor="w", padx=14).grid(row=row, column=0, sticky="ew",
                 pady=(8, 2)); row += 1

        mf = tk.Frame(sb, bg=PANEL)
        mf.grid(row=row, column=0, sticky="ew", padx=14, pady=2); row += 1

        tk.Label(mf, text="⎡  ⎤\n⎣  ⎦", bg=PANEL, fg=GRAY,
                 font=("Courier", 16)).grid(row=0, column=0, rowspan=2, padx=(0, 4))

        self.ev = {}
        for i, (lbl, val, r, c) in enumerate([
                ("a", "0",  0, 1),
                ("b", "-1", 0, 3),
                ("c", "-9", 1, 1),
                ("d", "0",  1, 3)]):
            tk.Label(mf, text=lbl+":", bg=PANEL, fg=TEXT_DIM,
                     font=self.f_small).grid(row=r, column=c-1, sticky="e", padx=2)
            var = tk.StringVar(value=val)
            _entry(mf, var).grid(row=r, column=c, padx=3, pady=3)
            var.trace_add("write", lambda *_: self._actualizar())
            self.ev[lbl] = var

        self._sep(row); row += 1

        # ── Vector b ──────────────────────────────────────────────────────────
        tk.Label(sb, text="▸ VECTOR  b",
                 bg=PANEL, fg=ORANGE, font=self.f_label,
                 anchor="w", padx=14).grid(row=row, column=0, sticky="ew",
                 pady=(8, 2)); row += 1

        vf = tk.Frame(sb, bg=PANEL)
        vf.grid(row=row, column=0, sticky="ew", padx=14, pady=2); row += 1

        tk.Label(vf, text="⎡ ⎤\n⎣ ⎦", bg=PANEL, fg=GRAY,
                 font=("Courier", 16)).grid(row=0, column=0, rowspan=2, padx=(0, 4))

        self.bv = {}
        for i, (lbl, val, r) in enumerate([("b₁", "1", 0), ("b₂", "9", 1)]):
            tk.Label(vf, text=lbl+":", bg=PANEL, fg=TEXT_DIM,
                     font=self.f_small).grid(row=r, column=1, sticky="e", padx=2)
            var = tk.StringVar(value=val)
            _entry(vf, var).grid(row=r, column=2, padx=3, pady=3)
            var.trace_add("write", lambda *_: self._actualizar())
            self.bv[lbl] = var

        self._sep(row); row += 1

        # ── Opciones ──────────────────────────────────────────────────────────
        tk.Label(sb, text="▸ VISUALIZACIÓN",
                 bg=PANEL, fg=ACCENT2, font=self.f_label,
                 anchor="w", padx=14).grid(row=row, column=0, sticky="ew",
                 pady=(8, 2)); row += 1

        opts = tk.Frame(sb, bg=PANEL)
        opts.grid(row=row, column=0, sticky="ew", padx=14); row += 1

        self.var_tray  = tk.BooleanVar(value=True)
        self.var_campo = tk.BooleanVar(value=True)
        self.var_evec  = tk.BooleanVar(value=True)
        self.var_null  = tk.BooleanVar(value=False)
        self.var_eq    = tk.BooleanVar(value=True)
        self.var_ejes  = tk.BooleanVar(value=True)

        for text, var in [
            ("Trayectorias",        self.var_tray),
            ("Campo vectorial",     self.var_campo),
            ("Autovectores",        self.var_evec),
            ("Nullclines",          self.var_null),
            ("Punto de equilibrio", self.var_eq),
            ("Ejes de coordenadas", self.var_ejes),
        ]:
            tk.Checkbutton(opts, text=text, variable=var,
                           bg=PANEL, fg=TEXT, selectcolor=BG,
                           activebackground=PANEL, activeforeground=ACCENT,
                           font=self.f_small, command=self._actualizar,
                           cursor="hand2").pack(anchor="w", pady=1)

        self._sep(row); row += 1

        # ── Rango ─────────────────────────────────────────────────────────────
        tk.Label(sb, text="▸ RANGO DEL PLANO",
                 bg=PANEL, fg=ACCENT2, font=self.f_label,
                 anchor="w", padx=14).grid(row=row, column=0, sticky="ew",
                 pady=(8, 2)); row += 1

        rf = tk.Frame(sb, bg=PANEL)
        rf.grid(row=row, column=0, sticky="ew", padx=14, pady=2); row += 1
        tk.Label(rf, text="Límite ±", bg=PANEL, fg=TEXT_DIM,
                 font=self.f_small).pack(side="left")
        self.var_rango = tk.StringVar(value="6")
        _entry(rf, self.var_rango, width=5).pack(side="left", padx=6)
        self.var_rango.trace_add("write", lambda *_: self._actualizar())

        self._sep(row); row += 1

        # ── Resultados ────────────────────────────────────────────────────────
        tk.Label(sb, text="▸ ANÁLISIS",
                 bg=PANEL, fg=ACCENT2, font=self.f_label,
                 anchor="w", padx=14).grid(row=row, column=0, sticky="ew",
                 pady=(8, 2)); row += 1

        res_outer = tk.Frame(sb, bg=BG,
                             highlightthickness=1, highlightbackground=BORDER)
        res_outer.grid(row=row, column=0, sticky="ew", padx=10, pady=2); row += 1

        self.lbl_res = tk.Label(res_outer, text="...",
                                bg=BG, fg=TEXT, font=self.f_small,
                                justify="left", padx=8, pady=6,
                                wraplength=280, anchor="w")
        self.lbl_res.pack(fill="x")

        self._sep(row); row += 1

        self.lbl_tipo = tk.Label(sb, text="",
                                 bg=PANEL, fg=GREEN, font=self.f_small,
                                 pady=4, wraplength=300, justify="center")
        self.lbl_tipo.grid(row=row, column=0, sticky="ew", padx=8); row += 1

        self._sep(row); row += 1

        tk.Button(sb, text="↺  RESTABLECER", command=self._reset,
                  bg=BORDER, fg=ACCENT, activebackground=ACCENT,
                  activeforeground=BG, font=self.f_label, relief="flat",
                  cursor="hand2", pady=5
                  ).grid(row=row, column=0, sticky="ew",
                         padx=12, pady=(4, 10)); row += 1

    def _sep(self, row):
        tk.Frame(self._sb, bg=BORDER, height=1).grid(
            row=row, column=0, sticky="ew", padx=8, pady=1)

    # ──────────────────────────────────────────────────────────────────────────
    # Gráfico
    # ──────────────────────────────────────────────────────────────────────────
    def _build_graph(self):
        gf = tk.Frame(self.root, bg=BG)
        gf.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        gf.rowconfigure(0, weight=1)
        gf.columnconfigure(0, weight=1)
        self.fig, self.ax = plt.subplots(figsize=(7.5, 6.5), facecolor=BG)
        self.ax.set_facecolor("#080b10")
        self.canvas = FigureCanvasTkAgg(self.fig, master=gf)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # ──────────────────────────────────────────────────────────────────────────
    # Lectura de entradas
    # ──────────────────────────────────────────────────────────────────────────
    _MATH_NS = {
        "__builtins__": {},
        "sin": np.sin,  "cos": np.cos,  "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp,  "log": np.log,  "log2": np.log2, "log10": np.log10,
        "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
        "pi": np.pi, "e": np.e, "inf": np.inf,
    }

    def _eval(self, expr: str):
        """Evalúa una expresión matemática segura y devuelve float, o None si hay error."""
        expr = expr.strip().replace("^", "**")
        try:
            val = float(eval(expr, self._MATH_NS))  # noqa: S307
            return val
        except Exception:
            return None

    def _get_Ab(self):
        vals = {}
        for k, var in {**self.ev, **self.bv}.items():
            v = self._eval(var.get())
            if v is None:
                return None, None
            vals[k] = v
        A    = np.array([[vals["a"], vals["b"]],
                         [vals["c"], vals["d"]]], dtype=float)
        bvec = np.array([vals["b₁"], vals["b₂"]], dtype=float)
        return A, bvec

    def _get_rango(self):
        try:
            return max(0.5, min(float(self.var_rango.get()), 30))
        except ValueError:
            return 6.0

    # ──────────────────────────────────────────────────────────────────────────
    # Actualización principal
    # ──────────────────────────────────────────────────────────────────────────
    def _actualizar(self):
        A, bvec = self._get_Ab()
        if A is None:
            return

        rango = self._get_rango()
        eigenvals, eigenvecs = np.linalg.eig(A)

        # Punto de equilibrio: Ax* = -b
        det = np.linalg.det(A)
        if abs(det) > 1e-12:
            x_eq = np.linalg.solve(A, -bvec)
            eq_str = f"x* = ({x_eq[0]:+.4f}, {x_eq[1]:+.4f})"
        else:
            x_eq = None
            eq_str = "x*: ∞ soluciones (det A = 0)"

        # Texto resultados
        texto = self._fmt(A, bvec, eigenvals, eigenvecs, det, eq_str)
        self.lbl_res.config(text=texto)

        tipo, color = self._clasificar(eigenvals)
        self.lbl_tipo.config(text=f"● {tipo}", fg=color)

        # Actualizar ecuación en header
        a, b = float(self.ev["a"].get()), float(self.ev["b"].get())
        c, d = float(self.ev["c"].get()), float(self.ev["d"].get())
        b1 = float(self.bv["b₁"].get()); b2 = float(self.bv["b₂"].get())
        self.lbl_ecuacion.config(
            text=f"ẋ = {a:+g}x {b:+g}y {b1:+g}\n"
                 f"ẏ = {c:+g}x {d:+g}y {b2:+g}")

        self._graficar(A, bvec, x_eq, eigenvals, eigenvecs, rango)

    def _fmt(self, A, bvec, eigenvals, eigenvecs, det, eq_str):
        tr = np.trace(A)
        lines = [
            f"tr(A)  = {tr:+.4f}",
            f"det(A) = {det:+.4f}",
            f"b      = ({bvec[0]:+g}, {bvec[1]:+g})",
            f"{eq_str}",
            "",
        ]
        for i, (λ, v) in enumerate(zip(eigenvals, eigenvecs.T)):
            if np.iscomplex(λ) and abs(λ.imag) > 1e-10:
                s = f"λ{i+1} = {λ.real:+.4f} {'+'if λ.imag>=0 else '-'} {abs(λ.imag):.4f}i"
                vr, vi = v.real, v.imag
                sv = (f"  v{i+1} = [{vr[0]:+.3f}{'+'if vi[0]>=0 else '-'}{abs(vi[0]):.3f}i,\n"
                      f"         {vr[1]:+.3f}{'+'if vi[1]>=0 else '-'}{abs(vi[1]):.3f}i]")
            else:
                s  = f"λ{i+1} = {λ.real:+.4f}"
                sv = f"  v{i+1} = [{v.real[0]:+.5f}, {v.real[1]:+.5f}]"
            lines += [s, sv, ""]
        return "\n".join(lines)

    def _clasificar(self, eigenvals):
        λ1, λ2 = eigenvals
        if np.iscomplex(λ1) and abs(λ1.imag) > 1e-10:
            re = λ1.real
            if abs(re) < 1e-10: return "Centro  (Im puros)",                ACCENT
            if re < 0:           return "Espiral estable  (Re < 0)",         GREEN
            return                      "Espiral inestable  (Re > 0)",        RED
        r1, r2 = λ1.real, λ2.real
        if r1 * r2 < 0:                return "Punto de silla",              YELLOW
        if r1 < 0 and r2 < 0:          return "Nodo estable",                GREEN
        if r1 > 0 and r2 > 0:          return "Nodo inestable",              RED
        if abs(r1) < 1e-10 or abs(r2) < 1e-10:
            return "Eigenvalor nulo — no aislado",                            ORANGE
        return "Tipo indeterminado",                                           GRAY

    # ──────────────────────────────────────────────────────────────────────────
    # Dibujo
    # ──────────────────────────────────────────────────────────────────────────
    def _graficar(self, A, bvec, x_eq, eigenvals, eigenvecs, rango):
        ax = self.ax
        ax.cla()
        ax.set_facecolor("#080b10")

        # Grilla del campo
        N = 22
        xs = np.linspace(-rango, rango, N)
        ys = np.linspace(-rango, rango, N)
        X, Y = np.meshgrid(xs, ys)
        # Sistema afín: ẋ = Ax + b
        U = A[0, 0]*X + A[0, 1]*Y + bvec[0]
        V = A[1, 0]*X + A[1, 1]*Y + bvec[1]

        # Campo vectorial
        if self.var_campo.get():
            speed = np.sqrt(U**2 + V**2)
            sn = speed / (speed.max() + 1e-10)
            ax.streamplot(X, Y, U, V, color=sn, cmap="cool",
                          linewidth=0.6, density=1.3, arrowsize=0.8)

        # Nullclines  (ẋ=0  y  ẏ=0, ahora son rectas afines)
        if self.var_null.get():
            xc = np.linspace(-rango, rango, 500)
            a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
            b1, b2 = bvec
            handles = []
            if abs(b) > 1e-12:
                yc1 = -(a*xc + b1) / b
                l1, = ax.plot(xc, yc1, color=ACCENT, lw=1.2, ls="--", alpha=0.7)
                handles.append((l1, "ẋ = 0"))
            if abs(d) > 1e-12:
                yc2 = -(c*xc + b2) / d
                l2, = ax.plot(xc, yc2, color=YELLOW, lw=1.2, ls="--", alpha=0.7)
                handles.append((l2, "ẏ = 0"))
            if handles:
                ax.legend([h for h,_ in handles], [n for _,n in handles],
                          fontsize=7, loc="upper right",
                          facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

        # Trayectorias (integradas desde el equilibrio desplazado)
        if self.var_tray.get():
            self._trayectorias(ax, A, bvec, x_eq, rango)

        # Autovectores (centrados en x*)
        if self.var_evec.get() and x_eq is not None:
            self._autovectores(ax, eigenvals, eigenvecs, x_eq, rango)

        # Punto de equilibrio
        if self.var_eq.get():
            if x_eq is not None:
                ax.plot(*x_eq, "o", color=RED, ms=7, zorder=15,
                        label=f"x*=({x_eq[0]:.2f},{x_eq[1]:.2f})")
                ax.annotate(f"x*=({x_eq[0]:.2f},{x_eq[1]:.2f})",
                            xy=x_eq,
                            xytext=(x_eq[0] + rango*0.05, x_eq[1] + rango*0.07),
                            color=RED, fontsize=7, fontfamily="monospace")
            # Origen (solo si b≠0, para referencia)
            if np.linalg.norm(bvec) > 1e-12:
                ax.plot(0, 0, "+", color=GRAY, ms=6, zorder=12)

        # Ejes y estilo
        if self.var_ejes.get():
            ax.axhline(0, color=GRAY, lw=0.7, alpha=0.6)
            ax.axvline(0, color=GRAY, lw=0.7, alpha=0.6)
        ax.set_xlim(-rango, rango)
        ax.set_ylim(-rango, rango)
        ax.set_xlabel("x", color=TEXT_DIM, fontsize=10)
        ax.set_ylabel("y", color=TEXT_DIM, fontsize=10, rotation=0, labelpad=12)
        ax.tick_params(colors=GRAY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.set_title("Diagrama de Fase  —  ẋ = Ax + b",
                     color=ACCENT, fontsize=11, fontfamily="monospace", pad=10)

        self.fig.tight_layout()
        self.canvas.draw()

    def _trayectorias(self, ax, A, bvec, x_eq, rango):
        from scipy.integrate import odeint

        def f(state, t):
            return A @ state + bvec

        t_fwd  = np.linspace(0,  14, 900)
        t_bwd  = np.linspace(0, -10, 500)
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        r      = rango * 0.82
        cmap   = plt.cm.plasma

        for i, ang in enumerate(angles):
            x0 = np.array([r * np.cos(ang), r * np.sin(ang)])
            col = cmap(i / len(angles))
            clip = rango * 2.5

            try:
                sol = odeint(f, x0, t_fwd)
                m = (np.abs(sol[:,0]) < clip) & (np.abs(sol[:,1]) < clip)
                ax.plot(sol[m, 0], sol[m, 1], color=col, lw=0.9, alpha=0.75)
            except Exception:
                pass
            try:
                sol = odeint(f, x0, t_bwd)
                m = (np.abs(sol[:,0]) < clip) & (np.abs(sol[:,1]) < clip)
                ax.plot(sol[m, 0], sol[m, 1], color=col, lw=0.8,
                        alpha=0.35, ls=":")
            except Exception:
                pass

    def _autovectores(self, ax, eigenvals, eigenvecs, x_eq, rango):
        colors_ev = [GREEN, RED]
        scale = rango * 0.6

        for i, (λ, col) in enumerate(zip(eigenvals, colors_ev)):
            v = eigenvecs[:, i]
            if np.iscomplex(λ) and abs(λ.imag) > 1e-10:
                continue
            vr = v.real
            norm = np.linalg.norm(vr)
            if norm < 1e-10:
                continue
            vr = vr / norm * scale
            # centrado en el equilibrio
            ox, oy = x_eq
            ax.annotate("", xy=(ox + vr[0], oy + vr[1]),
                        xytext=(ox - vr[0], oy - vr[1]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
            ax.text(ox + vr[0]*1.06, oy + vr[1]*1.06,
                    f"v{i+1}", color=col, fontsize=8, fontfamily="monospace")

    # ──────────────────────────────────────────────────────────────────────────
    def _reset(self):
        defs_e = {"a": "0", "b": "-1", "c": "-9", "d": "0"}
        defs_b = {"b₁": "1", "b₂": "9"}
        for k, v in defs_e.items(): self.ev[k].set(v)
        for k, v in defs_b.items(): self.bv[k].set(v)
        self.var_rango.set("6")
        self._actualizar()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    try:
        from scipy.integrate import odeint  # noqa: F401
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "scipy", "-q"])

    root = tk.Tk()
    SistemaDinamico2D(root)
    root.mainloop()


if __name__ == "__main__":
    main()