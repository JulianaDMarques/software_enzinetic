import sys
import numpy as np         # Biblioteca para cálculos numéricos
import pandas as pd        # Biblioteca para manipulação de dados
import matplotlib.pyplot as plt  # Biblioteca para plotar gráficos
from scipy.integrate import solve_ivp  # Resolve equações diferenciais

# ──────────────────────────────────────────────────────────────────────────────
#  MODELO PING-PONG BI-BI SEM k_-1 E k_-3, COM FATORES DE RENDIMENTO
# ──────────────────────────────────────────────────────────────────────────────

def modelo(Vmax, k1, k2, k3, k4, A, B):
    """
    Velocidade Ping-Pong Bi-Bi steady-state sem termos de retrocesso:
      Km_A = k2/k1, Km_B = k4/k3
      Vmax = E0 * min(k2, k4)
    """
    Km_A = k2 / k1
    Km_B = k4 / k3
    return (Vmax * A * B) / (Km_A * B + Km_B * A + A * B)

def f_conversao(s0_b, p_b):
    """Calcula a conversão percentual de S2 em P2."""
    return 100.0 * p_b / s0_b

def eq_dif(t, y, k1, k2, k3, k4, y1, y2):
    """
    Sistema de EDOs para o mecanismo Ping-Pong Bi-Bi:
    y = [E, S1, ES1, E_P1, S2, E_S2, P1, P2]
    Inclui fatores de rendimento y1 e y2.
    """
    E, S1, ES1, E_P1, S2, E_S2, P1, P2 = y

    dEdt    = -k1 * E * S1 + k4 * E_S2
    dS1dt   = -k1 * E * S1
    dES1dt  =  k1 * E * S1 - k2 * ES1
    dE_P1dt =  k2 * ES1 - k3 * E_P1 * S2
    dS2dt   = -k3 * E_P1 * S2
    dE_S2dt =  k3 * E_P1 * S2 - k4 * E_S2
    dP1dt   =  y1 * k2 * ES1        # rendimento no canal P1
    dP2dt   =  y2 * k4 * E_S2       # rendimento no canal P2

    return [dEdt, dS1dt, dES1dt, dE_P1dt, dS2dt, dE_S2dt, dP1dt, dP2dt]


# ──────────────────────────────────────────────────────────────────────────────
#  FUNÇÃO PARA PLOTAR GRÁFICOS DA SIMULAÇÃO
# ──────────────────────────────────────────────────────────────────────────────

def plot_sim(E0, Kcat_a, Kcat_b,
             s0_a, s0_b, t_input,
             k1, k2, k3, k4,
             y1=1.0, y2=1.0):
    """
    Gera quatro subplots e imprime k_app e ratio_p1p2 no console.
      1) Velocidade × Substrato
      2) Lineweaver–Burk
      3) Concentração × Tempo
      4) Conversão × Tempo
    Parâmetros adicionais:
      y1, y2 : fatores de rendimento para P1 e P2 (default = 1.0)
    """
    # ---- converte entradas para float ----
    E0      = float(E0)
    s0_a    = float(s0_a)
    s0_b    = float(s0_b)
    t_input = float(t_input)
    Kcat_a  = float(Kcat_a)
    Kcat_b  = float(Kcat_b)
    k1, k2, k3, k4 = map(float, (k1, k2, k3, k4))
    y1, y2        = float(y1), float(y2)

    # ---- métricas adicionais ----
    k_app      = min(k2, k4)
    ratio_p1p2 = (y1 * k2) / (y2 * k4) if (y2 * k4) != 0 else np.nan
    print(f"\n>> Apparent k (min(k2,k4)) = {k_app:.4g}")
    print(f">> Ratio P1/P2            = {ratio_p1p2:.4g}\n")

    # ---- prepara curvas de velocidade ----
    n_ptos = 100
    sa     = np.linspace(0, s0_a, n_ptos)
    sb     = np.linspace(0, s0_b, n_ptos)
    Vmax   = E0 * min(Kcat_a, Kcat_b)
    v      = modelo(Vmax, k1, k2, k3, k4, sa, sb)

    # ---- resolve o sistema de EDOs ----
    t   = np.linspace(0, t_input, n_ptos)
    y0  = [E0, s0_a, 0, 0, s0_b, 0, 0, 0]
    sol = solve_ivp(
        eq_dif,
        (t[0], t[-1]),
        y0,
        t_eval=t,
        args=(k1, k2, k3, k4, y1, y2),
        method='BDF'
    )
    y = sol.y

    # ---- conversão de S2 → P2 com rendimento ----
    conv_b = f_conversao(s0_b, y[7])

    # ---- preenche inversos para Lineweaver–Burk ----
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_v  = 1.0 / v
        inv_sa = 1.0 / sa
        inv_sb = 1.0 / sb

    # ---- desenha figura 2×2 ----
    plt.figure(figsize=(12, 8))

    # 1) Velocidade × Substrato
    plt.subplot(2, 2, 1)
    plt.plot(sa, v, '-',  color='firebrick',  label='S1 sim')
    plt.plot(sb, v, '--', color='darkorange', label='S2 sim')
    plt.title('Velocidade × Substrato')
    plt.xlabel('[S] (mol/L)'); plt.ylabel('V (mol/L·min)')
    plt.legend(); plt.grid()

    # 2) Lineweaver–Burk
    plt.subplot(2, 2, 2)
    plt.plot(inv_sa, inv_v, '-',  color='firebrick',  label='S1')
    plt.plot(inv_sb, inv_v, '--', color='darkorange', label='S2')
    plt.title('Lineweaver–Burk')
    plt.xlabel('1/[S]'); plt.ylabel('1/V')
    plt.legend(); plt.grid()

    # 3) Concentração × Tempo
    plt.subplot(2, 2, 3)
    plt.plot(t, y[1], '-',   color='firebrick',  label='S1 sim')
    plt.plot(t, y[4], '--',  color='darkorange', label='S2 sim')
    plt.plot(t, y[6], '-.',  color='navy',       label='P1 sim')
    plt.plot(t, y[7], ':',   color='dodgerblue', label='P2 sim')
    plt.title('Concentração × Tempo')
    plt.xlabel('Tempo (min)'); plt.ylabel('[C] (mol/L)')
    plt.legend(ncol=2); plt.grid()

    # 4) Conversão × Tempo
    plt.subplot(2, 2, 4)
    plt.plot(t, conv_b, '-', color='seagreen')
    plt.title('Conversão × Tempo')
    plt.xlabel('Tempo (min)'); plt.ylabel('Conversão (%)')
    plt.grid()

    plt.tight_layout()
    plt.show()
