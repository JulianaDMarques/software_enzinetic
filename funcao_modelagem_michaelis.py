import numpy as np  # Biblioteca para cálculos numéricos
import pandas as pd  # Biblioteca para manipulação de dados
import matplotlib.pyplot as plt  # Biblioteca para plotar gráficos
from scipy.integrate import solve_ivp  # Resolve equações diferenciais
from scipy.optimize import differential_evolution  # Ajuste de curvas
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

##Função de Modelo Michaelis Menten Sem Inibição                                  
def modelo(s,Vmax,Km):
    return Vmax*s/(Km+s) #Retorna Velocidade (V)

#Calculo da conversão
def f_conversao(s0, sf):
    return (sf/s0) * 100 # Calcula a conversão como percentual

#Equação diferencial -> Balanço de Massa
def eq_dif(t, C, km, Vmax):
    S, P = C 
    dSdt = - Vmax*S/(km + S)
    dPdt = Vmax*S/(km + S)

    return [dSdt, dPdt]



## Função que irá gerar a visualização final da tela e realizará todo o calculo da modelagem
# Essa função precisa do caminho da base de dados que o usuário irá inserir e do valor de E0
def funcao_final(file_path, E0, s0, adjustments, maxiter, popsize, mutation, recombination):

    #Função da leitura dos dados fornecidos pelo usuário
    def read_data_from_excel(file_path):
        """
        Leitura dos dados de concentração e tempo de substratos a partir de um arquivo Excel.
        """
        data = pd.read_excel(file_path, index_col=None, dtype={'Tempo': int,  'Produto': float})
        time = np.array(data['Tempo'].values, dtype = np.int32)
        substrate = s0
        produto = np.array(data['Produto'].fillna(0).values, dtype = np.float64)
        return time, substrate, produto

    time, substrate, produto = read_data_from_excel(file_path)
 
    # Função do modelo cinético
    def kinetic_model(t, k_params):
        """Simular os dados com os parâmetros cinéticos."""
        km, Vmax  = k_params
        k_params = (km, Vmax)
        y0 = [s0, produto[0]]
        sol = solve_ivp(eq_dif, [t[0], t[-1]], y0, t_eval=t, args=k_params, method='BDF') # Resolve EDO
        return sol ## Retorna vetores ao longo do tempo com o resultado da EDO

    # Ajuste dos parâmetros que serão usados no modelo 
    def ajustar_parametros(time, produto):
        def calculate_kinetics(time, produto):
            """Ajustar os parâmetros cinéticos com base nos dados experimentais."""
        
            def objetivo(params):
                km, Vmax = params
                k_params = (km, Vmax)
                sol = kinetic_model(time, k_params)
                
                errors = []
                if adjustments.get("S_adjust", False):
                    erro_S = np.sum((sol.y[0] - substrate) ** 2)
                    errors.append(erro_S)
                if adjustments.get("P_adjust", False):
                    erro_P = np.sum((sol.y[1] - produto) ** 2)
                    errors.append(erro_P)
                
                if errors:
                    return sum(errors)  
                else:
                    return 0

            bounds = [(0.001, 1000), (0.001, 1000)] # Limites para os parâmetros

            result = differential_evolution(objetivo, bounds, maxiter=maxiter, popsize=popsize, mutation=mutation, recombination=recombination, disp=True, polish=False)

            return result.x
    
        # Calcula Km e Vmax
        km, Vmax = calculate_kinetics(time, produto)
        Kcat = Vmax/E0
    
        return km, Vmax, Kcat

    # Encontrando as constantes necessárias
    km, Vmax, Kcat = ajustar_parametros(time, produto)
    

    # Plotagem dos resultados
    def plot_results(time, km, Vmax):
        """Gerar os gráficos com os resultados simulados e experimentais."""
        # Cria uma grade de tempo mais refinada para a simulação (delta menor)
        t_sim = np.linspace(time[0], time[-1], num=len(time) * 10)
        
        # Condições iniciais
        y0 = [s0, produto[0]]
        
        # Resolve a EDO utilizando a grade refinada t_sim
        sol = solve_ivp(
            eq_dif,
            [t_sim[0], t_sim[-1]],
            y0,
            t_eval=t_sim,
            args=(km,Vmax),
            method='BDF'
        )

        # Cálculo da Velocidade
        v = modelo(sol.y[0], Vmax, km)

        ## Gráfico 1: Velocidade x Substrato
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(sol.y[0], v, '-', color='firebrick', label='Substrato', markersize=3)
        plt.title('Velocidade x Substrato', fontsize=12, weight='bold')
        plt.xlabel('Concentração de Substrato (mol/L)', fontsize=10, weight='bold')
        plt.ylabel('Velocidade (mol/L.min)', fontsize=10, weight='bold')
        plt.legend(loc='best')
        plt.grid()

        ## Gráfico 2: Lineweaver-Burk
        inv_v = 1 / v
        inv_s = 1 / sol.y[0]
        
        plt.subplot(2, 2, 2)
        np.seterr(divide='ignore', invalid='ignore')
        plt.plot(inv_s, inv_v, 'o-', color='firebrick', label='Substrato', markersize=3)
        plt.title('Lineweaver-Burk Plot', fontsize=12, weight='bold')
        plt.xlabel('1/[Substrato] (1/mol/L)', fontsize=10, weight='bold')
        plt.ylabel('1/Velocidade (1/(mol/L.min)', fontsize=10, weight='bold')
        plt.legend(loc='best')
        plt.grid()

        ## Gráfico 3: Concentração x Tempo com eixo secundário
        ax1 = plt.subplot(2, 2, 3)
        # Eixo primário: Substrato 1 e Produto 1
        l1, = ax1.plot(t_sim, sol.y[0], ls='-', color='firebrick', label='Substrato')
        ax1.set_xlabel('Tempo (min)', fontsize=10, weight='bold')
        ax1.set_ylabel('Concentração (mol/L)', fontsize=10, weight='bold')

        # Eixo secundário: Substrato 2 e Produto 2
        ax2 = ax1.twinx()
        
        l4, = ax2.plot(t_sim, sol.y[1], ls=':', color='dodgerblue', label='Produto')
        sc1 = ax2.scatter(time, produto, color='navy', marker='o', label='Produto (experimental)')
        ax2.set_ylabel('Concentração (mol/L)', fontsize=10, weight='bold')

        # Combina as legendas dos dois eixos, incluindo os pontos experimentais
        lines = [l1, l4, sc1]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='best')
        ax1.grid(True)
        plt.title('Concentração x Tempo', fontsize=12, weight='bold')


        ## Gráfico 4: Conversão x Tempo
        plt.subplot(2, 2, 4)
        conversao = f_conversao(s0, sol.y[1])
        plt.plot(t_sim, conversao, '-', color='seagreen')
        plt.title('Conversão x Tempo', fontsize=12, weight='bold')
        plt.xlabel('Tempo (min)', fontsize=10, weight='bold')
        plt.ylabel('Conversão (%)', fontsize=10, weight='bold')
        plt.legend(loc='best')
        plt.grid()

        plt.tight_layout()
        plt.show()


    return ajustar_parametros(time, produto), plot_results(time, km, Vmax)


    

   

    