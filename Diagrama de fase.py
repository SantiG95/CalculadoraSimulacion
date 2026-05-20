"""
Diagrama de Fase 1D Autónomo
Modelos: Logístico, Enfriamiento de Newton y Fórmula Personalizada
Variables estándar: mu, K, h, X0
Requiere: pip install matplotlib numpy
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ── Paleta ───────────────────────────────────────────────────────────────
BG       = "#0f1117"
PANEL    = "#1a1d27"
ACCENT   = "#00e5ff"
ACCENT2  = "#ff4081"
ACCENT3  = "#b388ff"
TEXT     = "#e8eaf0"
MUTED    = "#6b7280"
ENTRY_BG = "#252836"
BORDER   = "#2d3148"
GREEN    = "#00e676"
RED      = "#ff1744"

FONT_TITLE = ("Courier New", 14, "bold")
FONT_LABEL = ("Courier New", 10)
FONT_SMALL = ("Courier New", 9)
FONT_BTN   = ("Courier New", 11, "bold")
FONT_MONO  = ("Courier New", 10)


# ── Modelos ──────────────────────────────────────────────────────────────

def _modelo_logistico(X, mu, K, **kw):
    """dX/dt = mu·X·(1 - X/K)"""
    return mu * X * (1.0 - X / K)

def _enfriamiento_newton(X, h, K, **kw):
    """dX/dt = -h·(X - K)   donde K = temperatura ambiente"""
    return -h * (X - K)


MODELOS = {
    "Modelo Logístico": {
        "func"     : _modelo_logistico,
        "ecuacion" : "dX/dt = mu · X · (1 - X/K)",
        "params"   : [
            ("mu  (tasa de crecimiento)", "mu",    "1.5"),
            ("K   (capacidad de carga)",  "K",     "10.0"),
            ("X_min",                     "X_min", "-2.0"),
            ("X_max",                     "X_max", "14.0"),
        ],
        "color"     : ACCENT,
        "puntos_eq" : lambda mu, K, **kw: [0.0, K],
        "desc"      : "Crecimiento poblacional con capacidad de carga K.\n"
                      "Equilibrios: X*=0 (inestable)  y  X*=K (estable).",
    },
    "Enfriamiento de Newton": {
        "func"     : _enfriamiento_newton,
        "ecuacion" : "dX/dt = -h · (X - K)",
        "params"   : [
            ("h   (constante de enfriamiento)", "h",     "0.5"),
            ("K   (temperatura ambiente)",      "K",     "20.0"),
            ("X_min",                           "X_min", "0.0"),
            ("X_max",                           "X_max", "100.0"),
        ],
        "color"     : ACCENT2,
        "puntos_eq" : lambda h, K, **kw: [K],
        "desc"      : "Temperatura de un objeto que se enfría hacia K.\n"
                      "Equilibrio: X*=K (estable).",
    },
    "Fórmula personalizada": {
        "func"      : None,
        "ecuacion"  : "dX/dt = f(X)",
        "params"    : [
            ("X_min", "X_min", "-5.0"),
            ("X_max", "X_max", "15.0"),
        ],
        "color"     : ACCENT3,
        "puntos_eq" : None,
        "desc"      : "Escribí tu propia función f(X).\n"
                      "Podés usar mu, K, h como parámetros\n"
                      "y funciones de numpy: sin, cos, exp, sqrt…",
    },
}


# ── App ───────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Diagrama de Fase 1D Autónomo")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("1150x760")
        self._param_vars    = {}
        self._param_entries = {}
        self._formula_var   = tk.StringVar(value="mu*X*(1 - X/K)")
        self._build_ui()
        self._on_model_change()

    # ────────────────────────────────────── Layout

    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG, pady=10)
        hdr.pack(fill="x", padx=20)
        tk.Label(hdr, text="⚙  DIAGRAMA DE FASE 1D AUTÓNOMO",
                 bg=BG, fg=ACCENT, font=FONT_TITLE).pack(side="left")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        self._left = tk.Frame(body, bg=PANEL, width=315,
                              highlightbackground=BORDER, highlightthickness=1)
        self._left.pack(side="left", fill="y", padx=(0, 12))
        self._left.pack_propagate(False)
        self._build_controls(self._left)

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self._build_plots(right)

    # ────────────────────────────────────── Controles

    def _build_controls(self, parent):
        pad = {"padx": 16, "pady": 6}

        tk.Label(parent, text="MODELO", bg=PANEL, fg=MUTED,
                 font=FONT_SMALL).pack(anchor="w", **pad)

        self._model_var = tk.StringVar(value=list(MODELOS.keys())[0])
        self._combo = ttk.Combobox(parent, textvariable=self._model_var,
                                   values=list(MODELOS.keys()),
                                   state="readonly", font=FONT_LABEL, width=27)
        self._combo.pack(anchor="w", padx=16)
        self._combo.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())
        self._style_combo()

        self._eq_lbl = tk.Label(parent, text="", bg=PANEL, fg=ACCENT,
                                font=("Courier New", 9, "italic"),
                                wraplength=270, justify="left")
        self._eq_lbl.pack(anchor="w", padx=16, pady=(4, 0))

        _sep(parent)

        tk.Label(parent, text="PARÁMETROS", bg=PANEL, fg=MUTED,
                 font=FONT_SMALL).pack(anchor="w", **pad)

        self._params_frame = tk.Frame(parent, bg=PANEL)
        self._params_frame.pack(fill="x", padx=16)

        # Bloque fórmula personalizada
        self._custom_frame = tk.Frame(parent, bg=PANEL)
        self._custom_frame.pack(fill="x", padx=16)

        self._free_frame = tk.Frame(parent, bg=PANEL)
        self._free_frame.pack(fill="x", padx=16)

        _sep(parent)

        tk.Label(parent, text="CONDICIONES INICIALES", bg=PANEL, fg=MUTED,
                 font=FONT_SMALL).pack(anchor="w", **pad)

        ic_frame = tk.Frame(parent, bg=PANEL)
        ic_frame.pack(fill="x", padx=16)
        for label, key, default in [
            ("X0  (lista, ej: 1, 5, 9)", "X0_list", "1.0, 5.0, 9.0"),
            ("t_max",                     "t_max",   "10.0"),
        ]:
            self._make_entry(ic_frame, label, key, default)

        _sep(parent)

        self._desc_lbl = tk.Label(parent, text="", bg=PANEL, fg=TEXT,
                                  font=FONT_SMALL, wraplength=270, justify="left")
        self._desc_lbl.pack(anchor="w", padx=16, pady=4)

        _sep(parent)

        tk.Button(parent, text="▶  GRAFICAR", bg=ACCENT, fg=BG,
                  font=FONT_BTN, relief="flat", cursor="hand2",
                  activebackground="#00b8d9", activeforeground=BG,
                  command=self._graficar).pack(fill="x", padx=16, pady=10)

        tk.Button(parent, text="↺  LIMPIAR", bg=ENTRY_BG, fg=TEXT,
                  font=FONT_SMALL, relief="flat", cursor="hand2",
                  activebackground=BORDER, activeforeground=ACCENT,
                  command=self._limpiar).pack(fill="x", padx=16, pady=(0, 10))

    def _build_plots(self, parent):
        self._fig = Figure(figsize=(8, 6), facecolor=BG)
        self._fig.subplots_adjust(hspace=0.48, top=0.93, bottom=0.10,
                                  left=0.10, right=0.97)
        self._ax_phase = self._fig.add_subplot(2, 1, 1)
        self._ax_traj  = self._fig.add_subplot(2, 1, 2)
        for ax in (self._ax_phase, self._ax_traj):
            _style_ax(ax)
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    # ────────────────────────────────────── Helpers UI

    def _style_combo(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TCombobox",
                    fieldbackground=ENTRY_BG, background=ENTRY_BG,
                    foreground=TEXT, selectbackground=ENTRY_BG,
                    selectforeground=ACCENT, arrowcolor=ACCENT,
                    bordercolor=BORDER)

    def _make_entry(self, parent, label, key, default):
        tk.Label(parent, text=label, bg=PANEL, fg=TEXT,
                 font=FONT_SMALL).pack(anchor="w", pady=(4, 0))
        var = tk.StringVar(value=default)
        self._param_vars[key] = var
        e = tk.Entry(parent, textvariable=var, bg=ENTRY_BG, fg=ACCENT,
                     insertbackground=ACCENT, font=FONT_LABEL,
                     relief="flat", bd=4,
                     highlightbackground=BORDER, highlightthickness=1)
        e.pack(fill="x", pady=(0, 2))
        self._param_entries[key] = e

    def _on_model_change(self):
        name = self._model_var.get()
        cfg  = MODELOS[name]
        self._eq_lbl.config(text=cfg["ecuacion"])
        self._desc_lbl.config(text=cfg["desc"])

        for frame in (self._params_frame, self._custom_frame, self._free_frame):
            for w in frame.winfo_children():
                w.destroy()

        keep = {"X0_list", "t_max"}
        self._param_vars    = {k: v for k, v in self._param_vars.items()    if k in keep}
        self._param_entries = {k: v for k, v in self._param_entries.items() if k in keep}

        for label, key, default in cfg["params"]:
            self._make_entry(self._params_frame, label, key, default)

        if name == "Fórmula personalizada":
            # Campo de expresión
            tk.Label(self._custom_frame, text="f(X) =", bg=PANEL, fg=ACCENT3,
                     font=FONT_SMALL).pack(anchor="w", pady=(8, 0))
            e = tk.Entry(self._custom_frame, textvariable=self._formula_var,
                         bg=ENTRY_BG, fg=ACCENT3, insertbackground=ACCENT3,
                         font=FONT_MONO, relief="flat", bd=4,
                         highlightbackground=ACCENT3, highlightthickness=1)
            e.pack(fill="x", pady=(0, 2))
            tk.Label(self._custom_frame,
                     text="Usá X como variable independiente.\nEjemplos:\n"
                          "  mu*X*(1-X/K)\n  -h*(X-K)\n  sin(X)-0.5",
                     bg=PANEL, fg=MUTED, font=("Courier New", 8),
                     justify="left").pack(anchor="w", pady=(2, 4))

            # Parámetros libres
            tk.Label(self._free_frame, text="PARÁMETROS LIBRES",
                     bg=PANEL, fg=MUTED, font=FONT_SMALL).pack(anchor="w",
                                                                pady=(6, 2))
            for label, key, default in [("mu", "mu", "1.5"),
                                         ("K",  "K",  "10.0"),
                                         ("h",  "h",  "0.5")]:
                self._make_entry(self._free_frame, label, key, default)

        self._limpiar()

    # ────────────────────────────────────── Parámetros

    def _get_params(self):
        vals = {}
        for key, var in self._param_vars.items():
            txt = var.get().strip()
            if key == "X0_list":
                vals[key] = [float(v) for v in txt.replace(";", ",").split(",")]
            else:
                vals[key] = float(txt)
        return vals

    def _build_custom_func(self, mu, K, h):
        expr = self._formula_var.get().strip()
        safe_ns = {fn: getattr(np, fn) for fn in dir(np) if not fn.startswith("_")}
        safe_ns.update({"np": np, "mu": mu, "K": K, "h": h})
        def f(X, **kw):
            local = dict(safe_ns)
            local["X"] = X
            return eval(expr, {"__builtins__": {}}, local)
        return f, expr

    def _find_equilibria_numeric(self, func, x_min, x_max):
        xs  = np.linspace(x_min, x_max, 2000)
        try:
            ys = func(xs)
        except Exception:
            return []
        eqs = []
        for i in range(len(ys) - 1):
            if np.isfinite(ys[i]) and np.isfinite(ys[i+1]):
                if ys[i] * ys[i+1] < 0:
                    a, b = xs[i], xs[i+1]
                    for _ in range(50):
                        m = (a + b) / 2
                        fm = func(np.array([m]))[0]
                        if func(np.array([a]))[0] * fm < 0:
                            b = m
                        else:
                            a = m
                    eqs.append((a + b) / 2)
        return eqs

    # ────────────────────────────────────── Graficación

    def _graficar(self):
        try:
            p = self._get_params()
        except ValueError as exc:
            messagebox.showerror("Error de parámetros", f"Valor inválido: {exc}")
            return

        name  = self._model_var.get()
        cfg   = MODELOS[name]
        color = cfg["color"]

        X_min   = p.get("X_min", -2.0)
        X_max   = p.get("X_max", 14.0)
        t_max   = p.get("t_max", 10.0)
        X0_list = p.get("X0_list", [1.0])

        # ── Función ─────────────────────────────────────────────────────
        if name == "Fórmula personalizada":
            mu = p.get("mu", 1.5)
            K  = p.get("K",  10.0)
            h  = p.get("h",  0.5)
            try:
                func, expr_str = self._build_custom_func(mu, K, h)
                func(np.array([1.0, 2.0]))   # validar
            except Exception as exc:
                messagebox.showerror("Error en fórmula",
                                     f"No se pudo evaluar f(X):\n{exc}")
                return
            kw_func   = {}
            eq_pts    = self._find_equilibria_numeric(func, X_min, X_max)
            title_str = f"f(X) = {expr_str}"
        else:
            func      = cfg["func"]
            kw_func   = {k: v for k, v in p.items()
                         if k not in ("X_min", "X_max", "t_max", "X0_list")}
            try:
                eq_pts = cfg["puntos_eq"](**kw_func)
            except Exception:
                eq_pts = []
            title_str = name

        # ── Vectores ────────────────────────────────────────────────────
        X  = np.linspace(X_min, X_max, 800)
        fX = func(X, **kw_func)

        # ── Diagrama de fase ────────────────────────────────────────────
        ax = self._ax_phase
        ax.cla()
        _style_ax(ax)

        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ax.plot(X, fX, color=color, lw=2.2, zorder=2)

        ax.fill_between(X, 0, fX, where=(fX > 0), alpha=0.13,
                        color=color, interpolate=True)
        ax.fill_between(X, 0, fX, where=(fX < 0), alpha=0.13,
                        color=ACCENT2, interpolate=True)

        # Flechas de flujo sobre el eje X
        step   = (X_max - X_min) / 18
        arr_xs = np.arange(X_min + step / 2, X_max, step)
        for xi in arr_xs:
            fi = float(np.atleast_1d(func(np.array([xi]), **kw_func))[0])
            if abs(fi) < 1e-9:
                continue
            dx = np.sign(fi) * (X_max - X_min) * 0.013
            ax.annotate("", xy=(xi + dx, 0), xytext=(xi, 0),
                        arrowprops=dict(arrowstyle="->",
                                        color=color if fi > 0 else ACCENT2,
                                        lw=1.3))

        # Puntos de equilibrio
        for xeq in eq_pts:
            eps  = 1e-5
            fxep = float(np.atleast_1d(func(np.array([xeq + eps]), **kw_func))[0])
            fxem = float(np.atleast_1d(func(np.array([xeq - eps]), **kw_func))[0])
            dfdx = (fxep - fxem) / (2 * eps)
            stable = dfdx < 0
            fc = GREEN if stable else RED
            ax.plot(xeq, 0, "o", markersize=11,
                    markerfacecolor=fc if stable else BG,
                    markeredgewidth=2.3, markeredgecolor=fc, zorder=5)
            ax.annotate(f"X*={xeq:.3g} {'●' if stable else '○'}",
                        (xeq, 0), textcoords="offset points", xytext=(4, 10),
                        color=fc, fontsize=8, fontfamily="Courier New")

        ax.set_title(f"{title_str}  —  Diagrama de Fase",
                     color=TEXT, fontsize=9, fontfamily="Courier New")
        ax.set_xlabel("X", color=MUTED, fontsize=9, fontfamily="Courier New")
        ax.set_ylabel("f(X) = dX/dt", color=MUTED, fontsize=9,
                      fontfamily="Courier New")

        pp = mpatches.Patch(color=color,   alpha=0.45, label="dX/dt > 0  →")
        pn = mpatches.Patch(color=ACCENT2, alpha=0.45, label="dX/dt < 0  ←")
        ps = mpatches.Patch(color=GREEN,               label="Estable  ●")
        pu = mpatches.Patch(color=RED,                 label="Inestable ○")
        ax.legend(handles=[pp, pn, ps, pu], loc="upper right",
                  framealpha=0.2, labelcolor=TEXT, fontsize=7.5,
                  facecolor=PANEL, edgecolor=BORDER)

        # ── Trayectorias X(t) ────────────────────────────────────────────
        ax2 = self._ax_traj
        ax2.cla()
        _style_ax(ax2)

        t  = np.linspace(0, t_max, 1000)
        dt = t[1] - t[0]
        cmap      = plt.cm.plasma
        colors_ic = [cmap(i / max(len(X0_list) - 1, 1))
                     for i in range(len(X0_list))]

        for X0, ci in zip(X0_list, colors_ic):
            xs = [X0]
            for _ in t[1:]:
                fv = float(np.atleast_1d(
                    func(np.array([xs[-1]]), **kw_func))[0])
                xs.append(xs[-1] + dt * fv)
            ax2.plot(t, xs, color=ci, lw=1.9, label=f"X0={X0:.3g}")

        for xeq in eq_pts:
            ax2.axhline(xeq, color="#90caf9", lw=0.8,
                        linestyle="--", alpha=0.6)
            ax2.text(t_max * 0.02, xeq, f"X*={xeq:.3g}",
                     color="#90caf9", fontsize=7.5,
                     fontfamily="Courier New", va="bottom")

        ax2.set_title("Trayectorias X(t)", color=TEXT, fontsize=9,
                      fontfamily="Courier New")
        ax2.set_xlabel("t", color=MUTED, fontsize=9, fontfamily="Courier New")
        ax2.set_ylabel("X(t)", color=MUTED, fontsize=9,
                       fontfamily="Courier New")
        ax2.legend(loc="upper right", framealpha=0.2, labelcolor=TEXT,
                   fontsize=7.5, facecolor=PANEL, edgecolor=BORDER)

        self._canvas.draw()

    def _limpiar(self):
        for ax in (self._ax_phase, self._ax_traj):
            ax.cla()
            _style_ax(ax)
        self._canvas.draw()


# ── Helpers matplotlib ────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor("#13161f")
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, linewidth=0.5, linestyle="--", alpha=0.6)

def _sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=10, pady=6)


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()