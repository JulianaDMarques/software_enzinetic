# Simulação e Modelagem de Reações Enzimáticas  
Interface gráfica em Python (PyQt5) para cinética **Ping Pong Bi‑Bi** e **Michaelis‑Menten**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![GUI](https://img.shields.io/badge/GUI-PyQt5-green)



---

## 🔬 Introdução teórica

### 1. Enzimas   
   As **enzimas** são macromoléculas proteicas que aceleram reações bioquímicas ao **reduzir a energia de ativação** do processo, sem alterar o equilíbrio químico global. Em virtude das enzimas, é importante compreender não apenas as suas propriedades intrínsecas, mas também os parâmetros cinéticos para analisar a velocidade das reações catalisadoras e os fatores químicos e físicos que podem afetar sua atividade (Johnson, 2021). Perante isto, Michaelis e Menten apresentaram um modelo para explicar como a velocidade das reações das enzimas, variam com a concentração de substrato. 

---

### 2. Mecanismo Cinética de Michaelis–Menten  
   Segundo este modelo, ilustrado pelo esquema abaixo, primeiramente o excesso de substrato (S) e a enzima (E) se ligam reversivelmente para formar o complexo enzima-substrato (ES), no processo seguinte, ocorre a separação do complexo e formação do produto (P) junto com a liberação da enzima livre (E).

O modelo elementar postula o ciclo:  

![](https://latex.codecogs.com/svg.image?E&plus;S\leftrightarrow&space;ES\rightarrow&space;E&plus;P)

A partir do esquema prévio, a seta na primeira etapa, indica a reversibilidade entre o complexo enzima-substrato (ES), e na segunda etapa, representa a taxa de moléculas de substrato convertidas em produto por molécula de enzima em unidade de tempo (Dixon; Webb, 1979). 

* Aplicando a **hipótese do estado estacionário** $(\(\frac{d[ES]}{dt}=0\))$ obtém‑se a equação de velocidade inicial:

$\[
v = \frac{V_\text{máx}\,[S]}{K_m + [S]}
\]$

| Constante | Definição | Interpretação |
|-----------|-----------|---------------|
| $\(V_\text{máx}=k_{2}[E]_0\)$ | Velocidade máxima | Condição de saturação $(\([S]\gg K_m\))$ |
| $\(K_m=\dfrac{k_{-1}+k_{2}}{k_{1}}\)$ | Constante de Michaelis | Medida inversa da afinidade $\(E\)–\(S\)$ |
| $\(k_\text{cat}=k_{2}\)$ | Constante catalítica | Nº de moléculas de produto geradas por enzima·s |

---

#### 2.1  Determinação experimental  
* **Ajuste não‑linear** (preferível): regressão direta da equação de Michaelis–Menten sobre \(v\) vs \([S]\).  
* **Linearizações clássicas** (ilustrativas):  

| Transformação | Equação | Gráfico |
|---------------|---------|---------|
| **Lineweaver–Burk** | $\(\frac{1}{v}=\frac{K_m}{V_\text{máx}}\frac{1}{[S]}+\frac{1}{V_\text{máx}}\)$ | Reta; intercepto $\(=1/V_\text{máx}\); inclinação \(=K_m/V_\text{máx}\)$ |
| **Eadie–Hofstee** | $\(v = -K_m\frac{v}{[S]} + V_\text{máx}\)$ | Reta em $\(v\)$ vs $\(v/[S]\)$ |
| **Hanes–Woolf** | $\(\frac{[S]}{v}=\frac{[S]}{V_\text{máx}}+\frac{K_m}{V_\text{máx}}\)$ | Reta em $\([S]/v\)$ vs $\([S]\)$ |

---

### 3. Mecanismo Ping Pong Bi‑Bi 
O método de Michaelis-Menten, proposto para reações com um único substrato, tornou-se base para estudos de novas técnicas de cinética enzimática, tais como o mecanismo de Ping-Pong Bi-Bi, que fornece uma metodologia mais ampliada de reações de multi-substrato em sistemas enzimáticos complexos (Gonçalves et al., 2021). Dito isso, abaixo, encontra-se esquematizado o funcionamento do mecanismo de Ping Pong Bi-Bi envolvendo reação com **dois substratos (A, B)** e **dois produtos (P, Q)**:

![](https://latex.codecogs.com/svg.image?E\overset{S_1\downarrow}{\rightarrow}ES_1\leftrightarrow&space;E'P_1\overset{P_1\uparrow}{\rightarrow}E'\overset{S_2\downarrow}{\rightarrow}E'S_2\leftrightarrow&space;EP_2\overset{P_2\uparrow}{\rightarrow}E)

Dado o esquema anterior, pode-se analisar que, o primeiro substrato se liga a
enzima (E), formando-se o primeiro complexo enzima-substrato ES1 (Ping), gerando, o
primeiro produto E′P1 (Bi). A enzima livre (E), no qual teve sua conformação modificada
(E′), liga-se a um segundo substrato (S2), formando um novo complexo enzima-substrato
ES2 (Pong) e por fim, a enzima volta para sua conformação inicial e o segundo produto
é liberado P2 (Bi)

A expressão de velocidade inicial (assumindo estado estacionário) é:

$\[
v = \frac{V_\text{máx}\,[A][B]}{K_{mB}[A] + K_{mA}[B] + [A][B]}
\]$

| Constante | Significado |
|-----------|-------------|
| $\(V_\text{máx}=k_{4}[E]_0\)$ | Velocidade máxima global |
| $\(K_{mA}\) e \(K_{mB}\)$ | Constantes de Michaelis aparentes para A e B |

---

#### 3.1  Características cinéticas  
* **Plots Lineweaver–Burk** de $\(1/v\)$ vs $\(1/[A]\)$ (com $\([B]\$) fixos) resultam em **retas paralelas**, diagnóstico típico de mecanismos Ping Pong.  
* Ajuste global é efetuado variando‑se simultaneamente $\([A]\)$ e $\([B]\)$ e minimizando o erro quadrático médio.

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



