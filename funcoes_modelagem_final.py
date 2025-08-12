# ──────────────────────────────────────────────────────────────────────────────
#  modelagem.py  –  Ping-Pong Bi-Bi with reversible binding (no yields)
# ──────────────────────────────────────────────────────────────────────────────
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QApplication
from PyQt5.QtCore import Qt
from scipy.interpolate import interp1d, make_interp_spline

# ──────────────────────────────────────────────────────────────────────────────
#  Cinética – equações auxiliares
# ──────────────────────────────────────────────────────────────────────────────
def modelo(Vmax, k1, k2, k3, k4, A, B):
    Km_A = k2 / k1          
    Km_B = k4 / k3          
    return (Vmax * A * B) / (Km_A * B + Km_B * A + A * B)

def f_conversao(s0_b: float, p_b: np.ndarray) -> np.ndarray:
    """
    Conversion [%] as product formed relative to initial substrate.
    """
    return 100.0 * p_b / s0_b

def eq_dif(t, y, k1, k_1, k2, k3, k_3, k4):
    """
    ODE system for Ping-Pong Bi-Bi with reversible substrate binding.
    """
    E, S1, ES1, E_P1, S2, E_S2, P1, P2 = y

    # Free enzyme
    dEdt    = -k1 * E * S1 + k_1 * ES1 \
              + k4 * E_S2

    # Substrate A
    dS1dt   = -k1 * E * S1 + k_1 * ES1

    # EA complex
    dES1dt  =  k1 * E * S1 - k_1 * ES1 - k2 * ES1

    # Enzyme–P1 intermediate
    dE_P1dt =  k2 * ES1 - k3 * E_P1 * S2 + k_3 * E_S2

    # Substrate B
    dS2dt   = -k3 * E_P1 * S2 + k_3 * E_S2

    # E*B complex
    dE_S2dt =  k3 * E_P1 * S2 - k_3 * E_S2 - k4 * E_S2

    # Products
    dP1dt   =  k2 * ES1
    dP2dt   =  k4 * E_S2

    return [dEdt, dS1dt, dES1dt, dE_P1dt, dS2dt, dE_S2dt, dP1dt, dP2dt]

# ──────────────────────────────────────────────────────────────────────────────
#  Janela de convergência (tempo-real)
# ──────────────────────────────────────────────────────────────────────────────
class ConvergenceDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Convergência — Differential Evolution")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.hist = []
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_xlabel("Geração")
        self.ax.set_ylabel("Função-objetivo")
        self.line, = self.ax.plot([], [], "-o", markersize=4, color="black")
        self.canvas = FigureCanvas(self.fig)
        lay = QVBoxLayout(); lay.addWidget(self.canvas); self.setLayout(lay)
        self.show()

    def update(self, value):
        self.hist.append(value)
        self.line.set_data(range(len(self.hist)), self.hist)
        self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw()
        QApplication.processEvents()

# ──────────────────────────────────────────────────────────────────────────────
#  Função principal chamada pela GUI
# ──────────────────────────────────────────────────────────────────────────────
def funcao_final(file_path: str,
                 E0: float,
                 s0_a: float, s0_b: float,
                 adjustments: dict,
                 maxiter: int, popsize: int,
                 mutation: tuple, recombination: float):

    # 1) Leitura dos dados
    dados = (pd.read_excel(file_path, dtype=str)
             .applymap(lambda x: str(x).replace(',', '.'))
             .astype(float))

    time_data   = dados["tempo"].values
    substrate_1 = dados["Substrato 1"].values
    substrate_2 = dados["Substrato 2"].values
    produto_1   = dados["Produto 1"].values
    produto_2   = dados["Produto 2"].values

    # 2) Setup do diálogo de convergência
    owns_qt = False
    if QApplication.instance() is None:
        owns_qt = True
        _ = QApplication(sys.argv)
    dlg = ConvergenceDialog()

    # 3) Solver da EDO com 7 parâmetros: (E0, k1, k_1, k2, k3, k_3, k4)
    def kinetic_model(t, p):
        e0, k1, k_1, k2, k3, k_3, k4 = p
        y0 = [
            e0,
            substrate_1[0], 0, 0,
            substrate_2[0], 0,
            produto_1[0], produto_2[0]
        ]
        return solve_ivp(
            eq_dif,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            args=(k1, k_1, k2, k3, k_3, k4),
            method="LSODA",
            rtol=1e-9,
            atol=1e-12
        )

    # 4) Função-objetivo + callback
    _EPS = 1e-6
    def _rmse_rel(sim, exp):
        rng = np.ptp(exp) + _EPS
        return np.sqrt(np.mean(((sim - exp) / rng) ** 2))

    def objective(p) -> float:
        try:
            sol = kinetic_model(time_data, p)
        except Exception:
            return 1e9
        errors = []
        if adjustments.get("S1_adjust"):
            errors.append(_rmse_rel(sol.y[1], substrate_1))
        if adjustments.get("S2_adjust"):
            errors.append(_rmse_rel(sol.y[4], substrate_2))
        if adjustments.get("P1_adjust"):
            errors.append(_rmse_rel(sol.y[6], produto_1))
        if adjustments.get("P2_adjust"):
            errors.append(_rmse_rel(sol.y[7], produto_2))
        if not errors:
            return 1e6
        return float(np.mean(errors))

    def de_callback(xk, convergence=None):
        dlg.update(objective(xk))
        return False

    # 5) Otimização – parâmetros de binding reversível
    best = dict(
        E0   = 4.995,
        k1   = 0.10,   k_1 = 0.010,
        k2   = 0.5048,
        k3   = 0.0904, k_3 = 0.00904,
        k4   = 0.10
    )
    bounds = [
        (max(1e-9, best['E0'] * 0.5),     min(10.0, best['E0'] * 1.5)),
        (max(1e-3, best['k1'] * 0.1),     min(5e3,  best['k1'] * 5.8)),
        (max(1e-3, best['k_1'] * 0.1),    min(5e3,  best['k_1'] * 5.8)),
        (max(1e-3, best['k2'] * 0.1),     min(5e3,  best['k2'] * 5.8)),
        (max(1e-3, best['k3'] * 0.1),     min(5e3,  best['k3'] * 5.8)),
        (max(1e-3, best['k_3'] * 0.1),    min(5e3,  best['k_3'] * 5.8)),
        (max(1e-3, best['k4'] * 0.1),     min(5e3,  best['k4'] * 5.8)),
    ]

    import time
    start_time = time.time()
    result = differential_evolution(
        objective, bounds,
        maxiter=maxiter, popsize=popsize,
        mutation=mutation, recombination=recombination,
        tol=1e-7, disp=False, polish=False,
        callback=de_callback
    )
    end_time = time.time()
    print(dlg.hist)
    print(f"Optimization completed in {end_time - start_time:.2f} seconds.")

    # 6) Desempacota parâmetros ajustados
    e0_opt, k1, k_1, k2, k3, k_3, k4 = result.x
    Km_A = (k_1 + k2) / k1
    Km_B = (k_3 + k4) / k3
    Vmax = e0_opt * min(k2, k4)
    params = (e0_opt, k1, k_1, k2, k3, k_3, k4, Km_A, Km_B, Vmax)

    # 7) R² das séries ajustadas
    sol_fit = kinetic_model(time_data, result.x)
    r2_map = {
        "S1_adjust": ("S1", r2_score(substrate_1, sol_fit.y[1])),
        "S2_adjust": ("S2", r2_score(substrate_2, sol_fit.y[4])),
        "P1_adjust": ("P1", r2_score(produto_1,   sol_fit.y[6])),
        "P2_adjust": ("P2", r2_score(produto_2,   sol_fit.y[7])),
    }

    # 7b) Estatísticas adicionais do modelo
    final_obj_value = objective(result.x)

    # RMSE absolutos
    def _rmse_abs(sim, exp):
        return np.sqrt(np.mean((sim - exp) ** 2))

    rmse_map = {
        "S1": _rmse_abs(sol_fit.y[1], substrate_1),
        "S2": _rmse_abs(sol_fit.y[4], substrate_2),
        "P1": _rmse_abs(sol_fit.y[6], produto_1),
        "P2": _rmse_abs(sol_fit.y[7], produto_2)
    }

    print("\n─── CONVERGÊNCIA FINAL ───")
    print(f"Final Objective Function: {final_obj_value:.6e}")
    print("\n─── TABELA DE ESTATÍSTICAS ───")

    data_table = {
        "Variável": [],
        "RMSE": [],
        "R²": []
    }

    for key, (label, r2) in r2_map.items():
        data_table["Variável"].append(label)
        data_table["RMSE"].append(f"{rmse_map[label]:.4e}")
        data_table["R²"].append(f"{r2:.4f}")

    summary_df = pd.DataFrame(data_table)
    print(summary_df.to_string(index=False))

    print("\n─── PARÂMETROS ADICIONAIS ───")
    print(f"Vmax           : {Vmax:.6f}")
    print(f"Km_A           : {Km_A:.6f}")
    print(f"Km_B           : {Km_B:.6f}")
    print(f"Apparent k_cat : {min(k2, k4):.6f}")


    # 8) Função de plotagem
    def plot_results(fig: plt.Figure | None = None):
        if fig is None:
            fig = plt.figure(figsize=(12, 8))
        fig.clf()
        gs = fig.add_gridspec(2, 2)

        # A) Velocidade × Substrato
        ax1 = fig.add_subplot(gs[0, 0])
        s1_range = np.linspace(sol_fit.y[1].min(), sol_fit.y[1].max(), 300)
        s2_fixed = sol_fit.y[4].mean()
        v_s1 = modelo(Vmax, k1, k2, k3, k4, s1_range, s2_fixed)

        s2_range = np.linspace(sol_fit.y[4].min(), sol_fit.y[4].max(), 300)
        s1_fixed = sol_fit.y[1].mean()
        v_s2 = modelo(Vmax, k1, k2, k3, k4, s1_fixed, s2_range)

        ax1.plot(s1_range, v_s1, '-',  label='S1 sim',   color='firebrick')
        ax1.plot(s2_range, v_s2, '--', label='S2 sim',   color='darkorange')
        ax1.scatter(substrate_1, [np.nan]*len(substrate_1),
                    marker='o', color='firebrick',  label='S1 exp')
        ax1.scatter(substrate_2, [np.nan]*len(substrate_2),
                    marker='s', color='darkorange', label='S2 exp')
        ax1.set(title='Velocidade × Substrato',
                xlabel='[S] (mol/L)', ylabel='V (mol/L·min)')
        ax1.legend(); ax1.grid(True)

        # B) Lineweaver–Burk
        ax2 = fig.add_subplot(gs[0, 1])
        inv_v_s1 = 1.0 / modelo(Vmax, k1, k2, k3, k4, sol_fit.y[1], s2_fixed)
        inv_v_s2 = 1.0 / modelo(Vmax, k1, k2, k3, k4, s1_fixed, sol_fit.y[4])
        ax2.plot(1/sol_fit.y[1], inv_v_s1, 'o-',  label='S1', color='firebrick')
        ax2.plot(1/sol_fit.y[4], inv_v_s2, 'o--', label='S2', color='darkorange')
        ax2.set(title='Lineweaver–Burk',
                xlabel='1/[S]', ylabel='1/V')
        ax2.legend(); ax2.grid(True)

        # C) Concentração × Tempo
        ax3 = fig.add_subplot(gs[1, 0])
        y0 = [
            e0_opt,
            substrate_1[0], 0, 0,
            substrate_2[0], 0,
            produto_1[0], produto_2[0]
        ]
        def ode_system(y, t):
            return eq_dif(t, y, k1, k_1, k2, k3, k_3, k4)

        sol2 = odeint(ode_system, y0, time_data,
                      rtol=1e-8, atol=1e-10)
        sol2 = sol2.T

        time_smooth = np.linspace(time_data[0], time_data[-1], 300)
        sol2_smooth = [make_interp_spline(time_data, sol2[i])(time_smooth)
                       for i in range(sol2.shape[0])]

        ax3.plot(time_smooth, sol2_smooth[1], '-',  color='firebrick',  label='S1 sim')
        ax3.plot(time_smooth, sol2_smooth[4], '--', color='darkorange', label='S2 sim')
        ax3.plot(time_smooth, sol2_smooth[6], '-.', color='navy',       label='P1 sim')
        ax3.plot(time_smooth, sol2_smooth[7], ':',  color='dodgerblue', label='P2 sim')
        ax3.scatter(time_data, substrate_1, marker='o', color='firebrick',  label='S1 exp')
        ax3.scatter(time_data, substrate_2, marker='s', color='darkorange', label='S2 exp')
        ax3.scatter(time_data, produto_1,   marker='^', color='navy',       label='P1 exp')
        ax3.scatter(time_data, produto_2,   marker='v', color='dodgerblue', label='P2 exp')
        ax3.set(title='Concentração × Tempo',
                xlabel='Tempo (min)', ylabel='[C] (mol/L)')
        ax3.legend(ncol=2); ax3.grid(True)

        # D) Conversão × Tempo
        ax4 = fig.add_subplot(gs[1, 1])
        conv = f_conversao(substrate_2[0], sol2[7])
        conv_interp = interp1d(time_data, conv, kind='cubic', fill_value="extrapolate")
        conv_smooth = conv_interp(time_smooth)
        ax4.plot(time_smooth, conv_smooth, '-', color='seagreen')
        ax4.set(title='Conversão × Tempo',
                xlabel='Tempo (min)', ylabel='Conversão (%)')
        ax4.grid(True)

        fig.tight_layout()
        return fig

    if owns_qt:
        QApplication.instance().exec_()
    return params, plot_results, r2_map
