#!/usr/bin/env python3
# coding: utf-8
"""
mm_model.py — Michaelis–Menten (1 → 1) kinetic fit
==================================================

Refatoração do código clássico de Michaelis–Menten para que adote a
mesma arquitetura empregada no modelo Ping‑Pong Bi‑Bi. Inclui:

• Integração completa em torno de *solve_ivp* (LSODA) para ajuste global
  Concentration × Time;
• Otimização por *differential_evolution* com janela de convergência em
  tempo‑real usando PyQt5;
• Função‑objetivo com RMSE relativo médio sobre Substrato e Produto
  (escala‑invariante);
• Estatísticas principais (RMSE, R²) em tabela impressa + valor final da
  função‑objetivo;
• Rotinas de plotagem: Velocidade × Substrato, Lineweaver–Burk,
  Concentration × Time, Conversão × Time.

Formato do arquivo de dados (CSV ou XLSX):
-----------------------------------------
Colunas obrigatórias (case‑insensitive):
    tempo, substrato, produto
Os valores podem usar vírgula decimal; a rotina converte internamente.

Uso:
-----
$ python mm_model.py dados.xlsx

Dependências: numpy, pandas, scipy, scikit‑learn, matplotlib, PyQt5.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout
from PyQt5.QtCore import Qt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score
from scipy.interpolate import make_interp_spline

###############################################################################
# Cinética — modelo Michaelis–Menten                                         #
###############################################################################

def v_mm(Vmax: float, Km: float, S: np.ndarray | float) -> np.ndarray | float:
    """Velocidade enzimática Michaelis–Menten sem inibição."""
    return Vmax * S / (Km + S)


def eq_dif(t: float, y: np.ndarray, Vmax: float, Km: float) -> list[float]:
    """Sistema de EDO para um substrato → um produto sob Michaelis–Menten."""
    S, P = y  # substrato, produto
    v = v_mm(Vmax, Km, S)
    return [-v, v]

###############################################################################
# Janela de convergência em tempo‑real                                       #
###############################################################################

class ConvergenceDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Convergência — Differential Evolution")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.history: list[float] = []
        self.figure, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_xlabel("Geração")
        self.ax.set_ylabel("Função‑objetivo (RMSE rel)")
        (self.line,) = self.ax.plot([], [], "-o", markersize=4, color="black")
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(); layout.addWidget(self.canvas); self.setLayout(layout)
        self.show()

    def update(self, value: float) -> None:
        self.history.append(value)
        self.line.set_data(range(len(self.history)), self.history)
        self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw()
        QApplication.processEvents()

###############################################################################
# Funções auxiliares                                                         #
###############################################################################

def _rmse_rel(sim: np.ndarray, exp: np.ndarray, eps: float = 1e-8) -> float:
    """RMSE relativo (escala‑invariante)."""
    return float(np.sqrt(np.mean(((sim - exp) / (np.ptp(exp) + eps)) ** 2)))


def _rmse_abs(sim: np.ndarray, exp: np.ndarray) -> float:
    """RMSE absoluto clássico."""
    return float(np.sqrt(np.mean((sim - exp) ** 2)))

###############################################################################
# Pipeline principal                                                         #
###############################################################################

def fit_mm(file_path: str | Path,
           maxiter: int = 5000,
           popsize: int = 20,
           mutation: tuple[float, float] = (0.5, 1.0),
           recombination: float = 0.7,
           ) -> tuple[dict[str, float], dict[str, float], dict[str, float], plt.Figure]:
    """Ajusta Vmax e Km a dados [S], [P] × tempo utilizando DE global.

    Retorna tupla: (params, rmse_map, r2_map, fig).
    """

    # 1) Leitura dos dados ----------------------------------------------------
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    if file_path.suffix.lower() in {".xls", ".xlsx"}:
        dados = pd.read_excel(file_path, dtype=str)
    else:
        dados = pd.read_csv(file_path, dtype=str)

    # normaliza nomes de colunas
    dados.columns = [c.strip().lower() for c in dados.columns]
    required = {"tempo", "substrato", "produto"}
    if not required.issubset(dados.columns):
        raise ValueError(f"Arquivo deve conter colunas: {', '.join(required)}")

    # converte vírgula decimal para ponto
    dados = dados.applymap(lambda x: str(x).replace(",", "."))
    dados = dados.astype(float)

    t_exp = dados["tempo"].values
    s_exp = dados["substrato"].values
    p_exp = dados["produto"].values

    # 2) Janela de convergência ----------------------------------------------
    owns_qt = False
    if QApplication.instance() is None:
        owns_qt = True
        _ = QApplication(sys.argv)
    dlg = ConvergenceDialog()

    # 3) Solver EDO encapsulado ----------------------------------------------

    def simulate(params: tuple[float, float]):
        Vmax, Km = params
        y0 = [s_exp[0], p_exp[0]]
        sol = solve_ivp(eq_dif,
                        (t_exp[0], t_exp[-1]), y0,
                        args=(Vmax, Km),
                        t_eval=t_exp, method="LSODA",
                        rtol=1e-9, atol=1e-12)
        if not sol.success:
            raise RuntimeError("Integration failed")
        return sol.y  # shape (2, len(t_exp))

    # 4) Função‑objetivo + callback ------------------------------------------
    def objective(x: np.ndarray) -> float:
        Vmax, Km = x
        if Vmax <= 0 or Km <= 0:
            return 1e8
        try:
            y_sim = simulate((Vmax, Km))
        except Exception:
            return 1e9
        err_s = _rmse_rel(y_sim[0], s_exp)
        err_p = _rmse_rel(y_sim[1], p_exp)
        return 0.5 * (err_s + err_p)

    def de_callback(xk: np.ndarray, convergence=None):
        dlg.update(objective(xk))
        return False

    # 5) Busca global (DE) ----------------------------------------------------
    bounds = [(max(1e-6, (p_exp[-1] - p_exp[0]) / (t_exp[-1] - t_exp[0])), 10.0 * (p_exp[-1] - p_exp[0]) / (t_exp[-1] - t_exp[0])),  # Vmax
              (1e-6, max(s_exp) * 10.0)]  # Km

    t_start = time.time()
    result = differential_evolution(objective, bounds,
                                    maxiter=maxiter, popsize=popsize,
                                    mutation=mutation, recombination=recombination,
                                    tol=1e-7, disp=False, polish=False,
                                    callback=de_callback)
    t_end = time.time()

    Vmax_opt, Km_opt = result.x

    # 6) Estatísticas de ajuste ----------------------------------------------
    y_best = simulate((Vmax_opt, Km_opt))
    r2_map = {
        "Substrato": r2_score(s_exp, y_best[0]),
        "Produto"  : r2_score(p_exp, y_best[1]),
    }
    rmse_map = {
        "Substrato": _rmse_abs(y_best[0], s_exp),
        "Produto"  : _rmse_abs(y_best[1], p_exp),
    }
    final_obj = objective(result.x)

    # 7) Impressão de relatórios ---------------------------------------------
    print("\n─── CONVERGÊNCIA FINAL ───")
    print(f"Função‑objetivo final: {final_obj:.6e}")
    print(f"Gerações totais      : {len(dlg.history)}")
    print(f"Tempo total          : {t_end - t_start:.2f} s")

    print("\n─── TABELA DE ESTATÍSTICAS ───")
    summary = pd.DataFrame({
        "Variável": list(r2_map.keys()),
        "RMSE": [f"{v:.4e}" for v in rmse_map.values()],
        "R²":   [f"{v:.4f}" for v in r2_map.values()],
    })
    print(summary.to_string(index=False))

    print("\n─── PARÂMETROS AJUSTADOS ───")
    print(f"Vmax : {Vmax_opt:.6e} (mol L⁻¹ min⁻¹)")
    print(f"Km   : {Km_opt:.6e} (mol L⁻¹)")

    # 8) Gráficos ------------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    # A) Velocidade × Substrato
    ax1 = fig.add_subplot(gs[0, 0])
    s_range = np.linspace(min(s_exp), max(s_exp), 300)
    v_range = v_mm(Vmax_opt, Km_opt, s_range)
    ax1.plot(s_range, v_range, '-', color='firebrick', label='Modelo')
    # pontos experimentais (velocidade inicial ~ dP/dt no 1º intervalo)
    v0_exp = np.gradient(p_exp, t_exp)
    ax1.scatter(s_exp, v0_exp, marker='o', color='navy', label='Exp')
    ax1.set(title='Velocidade × Substrato',
            xlabel='[S] (mol/L)', ylabel='V (mol/L·min)')
    ax1.grid(True); ax1.legend()

    # B) Lineweaver–Burk
    ax2 = fig.add_subplot(gs[0, 1])
    inv_v = 1.0 / v_mm(Vmax_opt, Km_opt, s_range)
    inv_s = 1.0 / s_range
    ax2.plot(inv_s, inv_v, '-', color='firebrick')
    ax2.scatter(1.0 / s_exp, 1.0 / v0_exp, marker='o', color='navy')
    ax2.set(title='Lineweaver–Burk', xlabel='1/[S]', ylabel='1/V')
    ax2.grid(True)

    # C) Concentração × Tempo
    ax3 = fig.add_subplot(gs[1, 0])
    time_smooth = np.linspace(t_exp[0], t_exp[-1], 300)
    sol_smooth = solve_ivp(eq_dif, (t_exp[0], t_exp[-1]), [s_exp[0], p_exp[0]],
                           t_eval=time_smooth, args=(Vmax_opt, Km_opt),
                           method='LSODA', rtol=1e-9, atol=1e-12)
    ax3.plot(time_smooth, sol_smooth.y[0], '-',  color='firebrick', label='S sim')
    ax3.plot(time_smooth, sol_smooth.y[1], '--', color='dodgerblue', label='P sim')
    ax3.scatter(t_exp, s_exp, marker='o', color='firebrick', label='S exp')
    ax3.scatter(t_exp, p_exp, marker='s', color='dodgerblue', label='P exp')
    ax3.set(title='Concentração × Tempo', xlabel='Tempo (min)', ylabel='[C] (mol/L)')
    ax3.grid(True); ax3.legend()

    # D) Conversão × Tempo
    ax4 = fig.add_subplot(gs[1, 1])
    conv = 100.0 * sol_smooth.y[1] / sol_smooth.y[0][0]
    conv_spline = make_interp_spline(time_smooth, conv)
    conv_dense = conv_spline(time_smooth)
    ax4.plot(time_smooth, conv_dense, '-', color='seagreen')
    ax4.set(title='Conversão × Tempo', xlabel='Tempo (min)', ylabel='Conversão (%)')
    ax4.grid(True)

    fig.tight_layout()

    if owns_qt:
        QApplication.instance().exec_()
    return {"Vmax": Vmax_opt, "Km": Km_opt}, rmse_map, r2_map, fig

###############################################################################
# Ponto de entrada                                                           #
###############################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python mm_model.py dados.[csv|xlsx]")
        sys.exit(1)
    fit_mm(sys.argv[1])
