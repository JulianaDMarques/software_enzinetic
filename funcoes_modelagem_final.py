import numpy as np  # Biblioteca para cálculos numéricos
import pandas as pd  # Biblioteca para manipulação de dados
import matplotlib.pyplot as plt  # Biblioteca para plotar gráficos
from scipy.integrate import solve_ivp  # Resolve equações diferenciais
from scipy.optimize import differential_evolution  # Ajuste de curvas
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#Função do modelo
def modelo(Vmax,k_1, k1, k2, k_3, k4, k3, A, B):
    Km_A = (k_1 + k2) / k1 # Cálculo de Km para substrato A
    Km_B = (k_3 + k4) / k3  # Cálculo de Km para substrato B
    v = (Vmax*A*B)/((Km_A*B) + (Km_B*A) + A*B) # Equação da velocidade
    return v

#Calculo da conversão
def f_conversao(s0_b, p_b):
    return ( p_b / s0_b) * 100 # Calcula a conversão como percentual

#Equação diferencial -> Balanço de Massa
def eq_dif(t, y, k1, k_1, k2, k3, k_3, k4):
    # Variáveis de estado (y): E, S1, ES1, E_P1, S2, E_S2, P1, P2
	E, S1, ES1, E_P1, S2, E_S2, P1, P2 = y

    # balanço de massa da reação enzimática ao longo do tempo
	dEdt = -k1 * E * S1 + k_1 * ES1 + k4 * E_S2
	dS1dt = -k1 * E * S1 + k_1 * ES1
	dES1dt = k1 * E * S1 - k_1 * ES1 - k2 * ES1
	dE_P1dt = k2 * ES1 - k3 * E_P1 * S2 + k_3 * E_S2
	dS2dt = -k3 * E_P1 * S2 + k_3 * E_S2
	dE_S2dt = k3 * E_P1 * S2 - k_3 * E_S2 - k4 * E_S2
	dP1dt = k2 * ES1
	dP2dt = k4 * E_S2

	return [dEdt, dS1dt, dES1dt, dE_P1dt, dS2dt, dE_S2dt, dP1dt, dP2dt] # Retorna as taxas de variação

## Função que irá gerar a visualização final da tela e realizará todo o calculo da modelagem
# Essa função precisa do caminho da base de dados que o usuário irá inserir e do valor de E0
def funcao_final(file_path, E0, s0_a, s0_b, adjustments, maxiter, popsize, mutation, recombination):

    #Função da leitura dos dados fornecidos pelo usuário
    def read_data_from_excel(file_path):
        """
        Leitura dos dados de concentração e tempo de substratos a partir de um arquivo Excel.
        """
        data = pd.read_excel(file_path, index_col=None, dtype={'tempo': int,  'Produto 1': float,  'Produto 2': float})
        time = np.array(data['tempo'].values, dtype = np.int32)
        #substrate_1 = data['Substrato 1'].fillna(0).values
        substrate_1 = s0_a
        substrate_2 = s0_b
        #substrate_2 = data['Substrato 2'].fillna(0).values
        produto_1 = np.array(data['Produto 1'].fillna(0).values, dtype = np.float64)
        produto_2 = np.array(data['Produto 2'].fillna(0).values, dtype = np.float64)
        return time, substrate_1, substrate_2, produto_1, produto_2

    time, substrate_1, substrate_2, produto_1, produto_2 = read_data_from_excel(file_path)
 
    # Função do modelo cinético
    def kinetic_model(t, k_params):
        """Simular os dados com os parâmetros cinéticos."""
        k1, k_1, k2, k3, k_3, k4 = k_params
        y0 = [E0, substrate_1, 0, 0, substrate_2, 0, produto_1[0], produto_2[0]] #condições iniciais 
        sol = solve_ivp(eq_dif, [t[0], t[-1]], y0, t_eval=t, args=k_params, method='BDF') # Resolve EDO
        return sol ## Retorna vetores ao longo do tempo com o resultado da EDO

    # Ajuste dos parâmetros que serão usados no modelo 
    def ajustar_parametros(time, produto_1, produto_2):
        def calculate_kinetics(time, produto_1, produto_2):
            """Ajustar os parâmetros cinéticos com base nos dados experimentais."""
                    
            # def objetivo(params):
            #     k1, k_1, k2, k3, k_3, k4 = params
            #     k_params = (k1, k_1, k2, k3, k_3, k4)
            #     sol = kinetic_model(time, k_params)
            #     erro_P1 = np.sum((sol.y[6] - produto_1) ** 2)
            #     erro_P2 = np.sum((sol.y[7] - produto_2) ** 2)
            #     return erro_P1 + erro_P2

            def objetivo(params):
                k1, k_1, k2, k3, k_3, k4 = params
                k_params = (k1, k_1, k2, k3, k_3, k4)
                sol = kinetic_model(time, k_params)
                
                errors = []
                if adjustments.get("S1_adjust", False):
                    erro_S1 = np.sum((sol.y[1] - substrate_1) ** 2)
                    errors.append(erro_S1)
                if adjustments.get("S2_adjust", False):
                    erro_S2 = np.sum((sol.y[4] - substrate_2) ** 2)
                    errors.append(erro_S2)
                if adjustments.get("P1_adjust", False):
                    erro_P1 = np.sum((sol.y[6] - produto_1) ** 2)
                    errors.append(erro_P1)
                if adjustments.get("P2_adjust", False):
                    erro_P2 = np.sum((sol.y[7] - produto_2) ** 2)
                    errors.append(erro_P2)
                
                if errors:
                    return sum(errors)  
                else:
                    return None

            bounds = [(0.001, 1000)] * 6  # Limites para os parâmetros

            result = differential_evolution(objetivo, bounds, maxiter=maxiter, popsize=popsize, mutation=mutation, recombination=recombination, disp=True, polish=False)

            return result.x
    
        # Calcula Km e Vmax
        k1, k_1, k2, k3, k_3, k4 = calculate_kinetics(time, produto_1, produto_2)
        Km_A = (k_1 + k2) / k1 # Cálculo de Km para substrato A
        Km_B = (k_3 + k4) / k3 # Cálculo de Km para substrato B
        Vmax = E0 * min(k2,k4) # Cálculo da Velocidade Máxima
    
        return k1, k_1, k2, k3, k_3, k4, Km_A, Km_B, Vmax

    # Encontrando as constantes necessárias
    k1, k_1, k2, k3, k_3, k4, Km_A, Km_B, Vmax = ajustar_parametros(time, produto_1, produto_2)
    

    # Plotagem dos resultados
    def plot_results(time, k1, k_1, k2, k3, k_3, k4):
        """Gerar os gráficos com os resultados simulados e experimentais."""
        # Cria uma grade de tempo mais refinada para a simulação (delta menor)
        t_sim = np.linspace(time[0], time[-1], num=len(time) * 10)
        
        # Condições iniciais
        y0 = [E0, substrate_1, 0, 0, substrate_2, 0, 0, 0]
        
        # Resolve a EDO utilizando a grade refinada t_sim
        sol = solve_ivp(
            eq_dif,
            [t_sim[0], t_sim[-1]],
            y0,
            t_eval=t_sim,
            args=(k1, k_1, k2, k3, k_3, k4),
            method='BDF'
        )

        # Cálculo da Velocidade
        v = (Vmax * sol.y[1] * sol.y[4]) / (sol.y[4] * Km_A + sol.y[1] * Km_B + sol.y[1] * sol.y[4])

        #Erros calculados 
        base1 = pd.DataFrame({'time': t_sim,  'produto2':sol.y[7]}).reset_index()
        base1['index'] = base1['index'] + 1
        base2 = pd.DataFrame({'time': time,  'produto2':produto_2}).reset_index()
        base2['index'] = base2['index'] + 1
        
        print("R² (pronounced r-squared):",r2_score(base2[base2['index'].isin([1,time[1:-1]])].produto2, base1[base1['index'].isin([1,time[1:-1]])].produto2))
        print("Mean squared error (MSE):",mean_squared_error(base2[base2['index'].isin([1,time[1:-1]])].produto2, base1[base1['index'].isin([1,time[1:-1]])].produto2))
        print("Mean absolute error (MAE):",mean_absolute_error(base2[base2['index'].isin([1,time[1:-1]])].produto2, base1[base1['index'].isin([1,time[1:-1]])].produto2))

        ## Gráfico 1: Velocidade x Substrato
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(sol.y[1], v, '-', color='firebrick', label='Substrato 1', markersize=3)
        plt.plot(sol.y[4], v, '--', color='darkorange', label='Substrato 2', markersize=3)
        plt.title('Velocidade x Substrato', fontsize=12, weight='bold')
        plt.xlabel('Concentração de Substrato (mol/L)', fontsize=10, weight='bold')
        plt.ylabel('Velocidade (mol/L.min)', fontsize=10, weight='bold')
        plt.legend(loc='best')
        plt.grid()

        ## Gráfico 2: Lineweaver-Burk
        inv_v = 1 / v
        inv_s1 = 1 / sol.y[1]
        inv_s2 = 1 / sol.y[4]
        
        plt.subplot(2, 2, 2)
        np.seterr(divide='ignore', invalid='ignore')
        plt.plot(inv_s1, inv_v, 'o-', color='firebrick', label='Substrato 1', markersize=3)
        plt.plot(inv_s2, inv_v, 'o--', color='darkorange', label='Substrato 2', markersize=3)
        plt.title('Lineweaver-Burk Plot', fontsize=12, weight='bold')
        plt.xlabel('1/[Substrato] (1/mol/L)', fontsize=10, weight='bold')
        plt.ylabel('1/Velocidade (1/(mol/L.min)', fontsize=10, weight='bold')
        plt.legend(loc='best')
        plt.grid()

        ## Gráfico 3: Concentração x Tempo com eixo secundário
        ax1 = plt.subplot(2, 2, 3)
        # Eixo primário: Substrato 1 e Produto 1
        l1, = ax1.plot(t_sim, sol.y[1], ls='-', color='firebrick', label='Substrato 1')
        l3, = ax1.plot(t_sim, sol.y[4], ls='--', color='darkorange', label='Substrato 2')
        ax1.set_xlabel('Tempo (min)', fontsize=10, weight='bold')
        ax1.set_ylabel('Concentração (mol/L)', fontsize=10, weight='bold')

        # Eixo secundário: Substrato 2 e Produto 2
        ax2 = ax1.twinx()
        
        l4, = ax2.plot(t_sim, sol.y[7], ls=':', color='dodgerblue', label='Produto 2')
        l2, = ax2.plot(t_sim, sol.y[6], ls='-.', color='navy', label='Produto 1')
        sc1 = ax2.scatter(time, produto_1, color='navy', marker='o', label='Produto 1 (experimental)')
        sc2 = ax2.scatter(time, produto_2, color='dodgerblue', marker='s', label='Produto 2 (experimental)')
        ax2.set_ylabel('Concentração (mol/L)', fontsize=10, weight='bold')

        # Combina as legendas dos dois eixos, incluindo os pontos experimentais
        lines = [l1, l2, l3, l4, sc1, sc2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='best')
        ax1.grid(True)
        plt.title('Concentração x Tempo', fontsize=12, weight='bold')


        ## Gráfico 4: Conversão x Tempo
        plt.subplot(2, 2, 4)
        conversao = 100 * sol.y[7] / substrate_2
        plt.plot(t_sim, conversao, '-', color='seagreen')
        plt.title('Conversão x Tempo', fontsize=12, weight='bold')
        plt.xlabel('Tempo (min)', fontsize=10, weight='bold')
        plt.ylabel('Conversão de Acetato de geranila (%)', fontsize=10, weight='bold')
        plt.legend(loc='best')
        plt.grid()

        plt.tight_layout()
        plt.show()


    return ajustar_parametros(time, produto_1, produto_2), plot_results(time, k1, k_1, k2, k3, k_3, k4)


    

   

    