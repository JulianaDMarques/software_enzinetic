import numpy as np  # Biblioteca para cálculos numéricos
import pandas as pd  # Biblioteca para manipulação de dados
import matplotlib.pyplot as plt  # Biblioteca para plotar gráficos
from scipy.integrate import solve_ivp  # Resolve equações diferenciais

#Função de Modelo Michaelis Menten Sem Inibição                                  
def modelo(s,Vmax,Km):
    return Vmax*s/(Km+s) #Retorna Velocidade (V)

#Calculo da conversão
def f_conversao(s0, sf):
    return (sf/s0) * 100 # Calcula a conversão como percentual

 #Equação diferencial -> Balanço de Massa
def eq_dif(t_input, C, km, E0, Kcat):
    s = C[0]
    p = C[1]

    dSdt = - (E0 * Kcat * s)/(km+s)
    dPdt = (E0 * Kcat * s)/(km+s)

    return [dSdt, dPdt]
    
#Função para plotar gráficos da simulação
def plot_sim(E0,Kcat,s0,p0,t_input,Km):
    
    #Variáveis modelo
    E0 = float(E0)
    s0 = float(s0)
    p0 = float(p0)
    t_input = float(t_input)
    Kcat = float(Kcat)
    Km = float(Km)
    

    # Criando o conjunto de dados artificiais -> ARRAY
    n_ptos = int(100)              #Número de pontos experimentais: 100 pontos
    s = np.linspace(0,s0, n_ptos)   #Gera um array com n_ptos de concentração de substrato 1 (s1) de 0 a sa_max
    Vmax = E0 * Kcat              #Cálculo da Velocidade Máxima

    v = modelo(s,Vmax, Km)   #Calcula a velocidade para cada s (concentração) do array através da função modelo
  

    #Concentração por Tempo
    t = np.linspace(0,t_input, n_ptos) #(0 a t_input minutos de reação, minuto a minuto)

    C0 = [s0,p0] #Concentração Inicial
    
    # Resolve EDO
    sol = solve_ivp(eq_dif, (t[0],t[-1]), C0, args=(Km,E0,Kcat), dense_output=True)

    # Gera os vetores para os gráficos
    y = sol.sol(t)

    #Calculo da conversão 
    conv = f_conversao(s0, y[1])

    with np.errstate(divide='ignore', invalid='ignore'):
        inv_v = 1 / v
        inv_s = 1 / s
    
# #Gráficos
    #Quadrante 1 -> Substratos pela velocidade
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(s, v, '-', color='firebrick', label='substrato', markersize=3)
    plt.title('Velocidade x Substrato', fontsize=12, weight='bold')
    plt.xlabel('Concentração de Substrato (mol/L)', fontsize=10, weight='bold')
    plt.ylabel('Velocidade (mol/L.min)', fontsize=10, weight='bold')
    plt.legend(loc='best')
    plt.grid()

    #Quadrante 2 -> 1/Velocidade x 1/Substratos
    plt.subplot(2, 2, 2)
    np.seterr(divide='ignore', invalid='ignore')
    plt.plot(inv_s,inv_v, 'o-', color = 'firebrick',label='substrato', markersize=3)
    plt.title('Lineweaver-Burk Plot', fontsize=12, weight='bold')
    plt.xlabel('1/[Substrato] (1/mol/L)', fontsize=10, weight='bold')
    plt.ylabel('1/Velocidade (1/(mol/L.min)', fontsize=10, weight='bold')
    plt.legend(loc='best')
    plt.grid()

    #Quadrante 3 -> Concentração (Substrato e Produto) x Tempo
    plt.subplot(2, 2, 3)
    plt.plot(t,y[0], ls='-', color='firebrick', label='Substrato')
    plt.plot(t,y[1], ls='-.', color='navy', label ='Produto')
    plt.title('Concentração x Tempo', fontsize=12, weight='bold')
    plt.xlabel('Tempo (min)', fontsize=10, weight='bold')
    plt.ylabel('Concentração (mol/L)', fontsize=10, weight='bold')
    plt.legend(loc='best')
    plt.grid()

    #Quadrante 4 -> Conversão x Tempo
    plt.subplot(2, 2, 4)
    plt.plot(t, conv, '-', color='seagreen')
    plt.title('Conversão x Tempo', fontsize=12, weight='bold')
    plt.xlabel('Tempo (min)', fontsize=10, weight='bold')
    plt.ylabel('Conversão (%)', fontsize=10, weight='bold')
    plt.legend(loc='best')
    plt.grid()

    plt.tight_layout()
    plt.show()