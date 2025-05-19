# Modelagem & Simulação de Reações Enzimáticas  
Interface gráfica em Python (PyQt5) para cinética **Ping Pong Bi‑Bi** e **Michaelis‑Menten**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![GUI](https://img.shields.io/badge/GUI-PyQt5-lightgrey)

---

## ✨ Visão geral
Este projeto oferece uma aplicação desktop que:

* **Simula** séries temporais de concentrações, velocidades e conversões.  
* **Ajusta** parâmetros cinéticos a dados experimentais via *Differential Evolution*.  
* Exibe **gráficos interativos** e exporta resultados diretamente na interface.

A GUI principal encontra‑se em **`tela_final_MM.py`**, a qual carrega dinamicamente módulos especializados para cada mecanismo:  

| Módulo | Papel | Mecanismo |
| ------ | ----- | --------- |
| `funcao_simulacao_final.py` | Simulação | Ping Pong Bi‑Bi |
| `funcoes_modelagem_final.py` | Modelagem | Ping Pong Bi‑Bi |
| `funcao_simulacao_michaelis.py` | Simulação | Michaelis‑Menten |
| `funcao_modelagem_michaelis.py` | Modelagem | Michaelis‑Menten |

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
* Métricas de qualidade (R², MSE, MAE) impressas no terminal.

---

## 🤝 Contribuindo
1. Faça um *fork* do projeto.  
2. Crie uma branch: `git checkout -b feature/nova-funcionalidade`  
3. *Commit* suas mudanças.  
4. Envie um *pull request*.

---

## 📝 Licença
Distribuído sob a licença **MIT**. Consulte `LICENSE` para mais detalhes.


