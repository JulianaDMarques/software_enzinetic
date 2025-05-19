# Modelagem & Simulação de Reações Enzimáticas  
Interface gráfica em Python (PyQt5) para cinética **Ping Pong Bi‑Bi** e **Michaelis‑Menten**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![GUI](https://img.shields.io/badge/GUI-PyQt5-green)



---

## 🔬 Introdução teórica

### 1. Enzimas   
As **enzimas** são macromoléculas proteicas que aceleram reações bioquímicas ao **reduzir a energia de ativação** do processo, sem alterar o equilíbrio químico global. Sua especificidade decorre da complementaridade entre o sítio ativo enzimático e o(s) substrato(s), permitindo estabilização transitória do **complexo enzima–substrato (ES)**.

### 2. Cinética de Michaelis–Menten  
O modelo elementar postula o ciclo:  

\[
E + S \;\xrightleftharpoons[k_{-1}]{k_{1}}\; ES \;\xrightarrow{k_{2}}\; E + P
\]

Aplicando a **hipótese do estado estacionário** (\(\frac{d[ES]}{dt}=0\)) obtém‑se a equação de velocidade inicial:

\[
v = \frac{V_\text{máx}\,[S]}{K_m + [S]}
\]

| Constante | Definição | Interpretação |
|-----------|-----------|---------------|
| \(V_\text{máx}=k_{2}[E]_0\) | Velocidade máxima | Condição de saturação (\([S]\gg K_m\)) |
| \(K_m=\dfrac{k_{-1}+k_{2}}{k_{1}}\) | Constante de Michaelis | Medida inversa da afinidade \(E\)–\(S\) |
| \(k_\text{cat}=k_{2}\) | Constante catalítica | Nº de moléculas de produto geradas por enzima·s |

#### 2.1  Determinação experimental  
* **Ajuste não‑linear** (preferível): regressão direta da equação de Michaelis–Menten sobre \(v\) vs \([S]\).  
* **Linearizações clássicas** (ilustrativas):  

| Transformação | Equação | Gráfico |
|---------------|---------|---------|
| **Lineweaver–Burk** | \(\frac{1}{v}=\frac{K_m}{V_\text{máx}}\frac{1}{[S]}+\frac{1}{V_\text{máx}}\) | Reta; intercepto \(=1/V_\text{máx}\); inclinação \(=K_m/V_\text{máx}\) |
| **Eadie–Hofstee** | \(v = -K_m\frac{v}{[S]} + V_\text{máx}\) | Reta em \(v\) vs \(v/[S]\) |
| **Hanes–Woolf** | \(\frac{[S]}{v}=\frac{[S]}{V_\text{máx}}+\frac{K_m}{V_\text{máx}}\) | Reta em \([S]/v\) vs \([S]\) |

### 3. Mecanismo Ping Pong Bi‑Bi (Duplo Deslocamento)  
Reação geral envolvendo **dois substratos (A, B)** e **dois produtos (P, Q)**:

\[
\begin{aligned}
E + A &\xrightleftharpoons[k_{-1}]{k_{1}} EA \xrightarrow{k_{2}} E^\* + P \\
E^\* + B &\xrightleftharpoons[k_{-3}]{k_{3}} E^\*B \xrightarrow{k_{4}} E + Q
\end{aligned}
\]

A enzima alterna entre as formas **\(E\)** e **\(E^\*\)**, liberando P antes de se ligar a B—dando origem ao termo *ping‑pong*.

A expressão de velocidade inicial (assumindo estado estacionário) é:

\[
v = \frac{V_\text{máx}\,[A][B]}{K_{mB}[A] + K_{mA}[B] + [A][B]}
\]

| Constante | Significado |
|-----------|-------------|
| \(V_\text{máx}=k_{4}[E]_0\) | Velocidade máxima global |
| \(K_{mA}\) e \(K_{mB}\) | Constantes de Michaelis aparentes para A e B |

#### 3.1  Características cinéticas  
* **Plots Lineweaver–Burk** de \(1/v\) vs \(1/[A]\) (com \([B]\) fixos) resultam em **retas paralelas**, diagnóstico típico de mecanismos Ping Pong.  
* Ajuste global é efetuado variando‑se simultaneamente \([A]\) e \([B]\) e minimizando o erro quadrático médio.

---
## ✨ Visão geral do software
A aplicação fornece: simulação temporal de concentrações, ajuste paramétrico via **Differential Evolution** e visualizações interativas. O módulo principal **`tela_final_MM.py`** gerencia a interface e carrega dinamicamente os algoritmos:

| Módulo | Papel | Mecanismo |
| ------ | ----- | --------- |
| `funcao_simulacao_final.py` | Simulação numérica | Ping Pong Bi‑Bi |
| `funcoes_modelagem_final.py` | Ajuste paramétrico | Ping Pong Bi‑Bi |
| `funcao_simulacao_michaelis.py` | Simulação numérica | Michaelis–Menten |
| `funcao_modelagem_michaelis.py` | Ajuste paramétrico | Michaelis–Menten |

---

## 📂 Estrutura do repositório
```
├── data/                     # Exemplos de planilhas .xlsx
├── images/                   # Logos e ícones para a GUI
├── funcao_simulacao_final.py
├── funcoes_modelagem_final.py
├── funcao_simulacao_michaelis.py
├── funcao_modelagem_michaelis.py
├── tela_final_MM.py          # Interface principal
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalação
1. Clone o repositório  
   ```bash
   git clone https://github.com/seu-usuario/seu-repo.git
   cd seu-repo
   ```

2. (Opcional) Crie um ambiente virtual  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. Instale as dependências  
   ```bash
   pip install -r requirements.txt
   ```

**`requirements.txt` (sugestão)**  
```text
PyQt5>=5.15
numpy
pandas
matplotlib
scipy
scikit-learn
```

---

## 🚀 Executando a aplicação
```bash
python tela_final_MM.py
```

---

## 🖥️ Guia de uso rápido

### 1 — Tela inicial  
* Selecione o mecanismo desejado *(Ping Pong Bi‑Bi ou Michaelis‑Menten)*.  
* Clique em **“Ir para Simulação”** ou **“Ir para Modelagem”**.

### 2 — Simulação  
1. Preencha os **parâmetros cinéticos** e **concentrações iniciais**.  
2. Clique em **“Gerar Simulação”** para visualizar quatro gráficos:  
   * Velocidade × Substrato  
   * Lineweaver‑Burk  
   * Concentração × Tempo  
   * Conversão × Tempo

### 3 — Modelagem (ajuste a dados experimentais)  
1. Clique em **“Selecionar Arquivo Excel”** e escolha sua planilha.  
   * O Excel deve conter as colunas:  

     **Michaelis‑Menten**  
     | Tempo | Produto |
     |-------|---------|

     **Ping Pong Bi‑Bi**  
     | tempo | Produto 1 | Produto 2 |
     |-------|-----------|-----------|

2. Insira **E0** e demais concentrações iniciais.  
3. (Opcional) Ajuste *maxiter*, *popsize*, *mutation* e *recombination*.  
4. Marque quais séries experimentais serão usadas como **“Ajustes Disponíveis”**.  
5. Clique em **“Gerar Modelagem”**.  
   * Os parâmetros otimizados aparecem no painel lateral.  
   * Os gráficos simulados × experimentais são atualizados automaticamente.

---

## 📈 Saída de resultados
* **Parâmetros estimados** (Km, Vmax, Kcat ou constantes k₁…k₄).  
* **Gráficos** salvos via botão direito (Matplotlib).  



