import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QPushButton, QWidget, QCheckBox,
    QLabel, QLineEdit, QMessageBox, QStackedWidget, QHBoxLayout, QGroupBox, QFrame, QTextEdit, QRadioButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QIcon
import importlib.util
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Importa as funções para Ping Pong Bi-Bi
simulacao_path = "funcao_simulacao_final.py"
modelagem_path = "funcoes_modelagem_final.py"

spec_sim = importlib.util.spec_from_file_location("funcoes_simulacao", simulacao_path)
funcoes_simulacao = importlib.util.module_from_spec(spec_sim)
spec_sim.loader.exec_module(funcoes_simulacao)

spec_mod = importlib.util.spec_from_file_location("funcoes_modelagem", modelagem_path)
funcoes_modelagem = importlib.util.module_from_spec(spec_mod)
spec_mod.loader.exec_module(funcoes_modelagem)

# Importa as funções para Michaelis Menten
simulacao_path_mm = "funcao_simulacao_michaelis.py"
modelagem_path_mm = "funcao_modelagem_michaelis.py"

spec_sim_mm = importlib.util.spec_from_file_location("funcoes_simulacao_mm", simulacao_path_mm)
funcoes_simulacao_mm = importlib.util.module_from_spec(spec_sim_mm)
spec_sim_mm.loader.exec_module(funcoes_simulacao_mm)

spec_mod_mm = importlib.util.spec_from_file_location("funcao_modelagem_mm", modelagem_path_mm)
funcoes_modelagem_mm = importlib.util.module_from_spec(spec_mod_mm)
spec_mod_mm.loader.exec_module(funcoes_modelagem_mm)


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modelagem e Simulação de Reações")
        self.setGeometry(300, 150, 500, 800)
        self.setFixedSize(1000, 1100)
        self.setWindowIcon(QIcon("imagem_unesp.png"))
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f7f9fc;
            }
            QLabel {
                font-size: 14px;
                color: #333333;
            }
            QPushButton {
                background-color: #2E86C1;
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1B4F72;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #bfc9ca;
                border-radius: 8px;
            }
            QGroupBox {
                border: 2px solid #5dade2;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 5px;
                font-size: 14px;
                font-weight: bold;
                color: #5dade2;
            }
        """)

        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.main_menu_ui()
        self.simulation_ui()
        self.modeling_ui()
        self.central_widget.setCurrentIndex(0)

    def main_menu_ui(self):
        """Tela inicial"""
        main_widget = QWidget()
        layout = QVBoxLayout()

        # Logo e título
        #logo = QLabel()
        #pixmap = QPixmap("imagem_unesp.png").scaled(100, 100, Qt.KeepAspectRatio)
        #logo.setPixmap(pixmap)
        #logo.setAlignment(Qt.AlignCenter)

        title = QLabel("Modelagem e Simulação de Reações")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2E4053;")

        # Escolha da Metodologia
        methodology_group = QGroupBox("Escolha a Metodologia")
        methodology_layout = QHBoxLayout()
        self.method_pingpong = QRadioButton("Ping Pong Bi-Bi")
        self.method_michaelis = QRadioButton("Michaelis Menten")
        self.method_pingpong.setChecked(True)
        methodology_layout.addWidget(self.method_pingpong)
        methodology_layout.addWidget(self.method_michaelis)
        methodology_group.setLayout(methodology_layout)

        button_frame = QFrame()
        button_layout = QVBoxLayout()

        # Funções que atualizam os campos antes de mudar de página
        simulation_button = QPushButton("Ir para Simulação")
        #simulation_button.setIcon(QIcon.fromTheme("media-playback-start"))
        simulation_button.clicked.connect(self.go_to_simulation)

        modeling_button = QPushButton("Ir para Modelagem")
        #modeling_button.setIcon(QIcon.fromTheme("document-open"))
        modeling_button.clicked.connect(self.go_to_modeling)

        button_layout.addWidget(simulation_button)
        button_layout.addWidget(modeling_button)
        button_layout.setSpacing(20)
        button_frame.setLayout(button_layout)

        #layout.addWidget(logo)
        layout.addWidget(title)
        layout.addWidget(methodology_group)
        layout.addStretch()
        layout.addWidget(button_frame)
        layout.addStretch()

        main_widget.setLayout(layout)
        self.central_widget.addWidget(main_widget)

    def go_to_simulation(self):
        self.update_simulation_ui()
        self.central_widget.setCurrentIndex(1)

    def go_to_modeling(self):
        self.update_modeling_ui()
        self.central_widget.setCurrentIndex(2)

    def update_simulation_ui(self):
        """Atualiza os campos da simulação conforme o método escolhido."""
        if self.method_michaelis.isChecked():
            self.mm_input_group.setVisible(True)
            self.input_group.setVisible(False)
        else:
            self.mm_input_group.setVisible(False)
            self.input_group.setVisible(True)

    def update_modeling_ui(self):
        """Atualiza os campos da modelagem conforme o método escolhido."""
        if self.method_michaelis.isChecked():
            self.mm_conc_group.setVisible(True)
            self.conc_group.setVisible(False)
            self.check_group_mm.setVisible(True)
            self.check_group_pingpong.setVisible(False)
        else:
            self.mm_conc_group.setVisible(False)
            self.conc_group.setVisible(True)
            self.check_group_mm.setVisible(False)
            self.check_group_pingpong.setVisible(True)

    def simulation_ui(self):
        """Tela de Simulação"""
        sim_widget = QWidget()
        layout = QVBoxLayout()

        header = QLabel("Simulação de Reações")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Grupo para simulação Ping Pong Bi-Bi
        self.input_group = QGroupBox("Parâmetros da Simulação (Ping Pong Bi-Bi)")
        input_layout = QVBoxLayout()
        self.inputs = {}
        parametros = [
            ("E0", "Concentração Inicial da Enzima (mol/L)", "0.01"),
            ("Kcat_a", "Kcat do Substrato A", "100"),
            ("Kcat_b", "Kcat do Substrato B", "100"),
            ("s0_a", "Concentração Inicial Substrato A (mol/L)", "0.1"),
            ("s0_b", "Concentração Inicial Substrato B (mol/L)", "0.1"),
            ("t_input", "Tempo de Simulação (min)", "60"),
            ("k1", "Constante k1", "1.0"),
            ("k2", "Constante k2", "1.0"),
            ("k3", "Constante k3", "1.0"),
            ("k4", "Constante k4", "1.0")
        ]
        for key, label, default in parametros:
            row = QHBoxLayout()
            lbl = QLabel(label)
            input_field = QLineEdit()
            input_field.setText(default)
            self.inputs[key] = input_field
            row.addWidget(lbl)
            row.addWidget(input_field)
            input_layout.addLayout(row)
        self.input_group.setLayout(input_layout)
        layout.addWidget(self.input_group)

        # Grupo para simulação Michaelis Menten
        self.mm_input_group = QGroupBox("Parâmetros da Simulação (Michaelis Menten)")
        mm_input_layout = QVBoxLayout()
        self.inputs_mm = {}
        mm_parametros = [
            ("E0", "Concentração Inicial da Enzima (mol/L)", "0.01"),
            ("Kcat", "Kcat", "100"),
            ("s0", "Concentração Inicial do Substrato (mol/L)", "0.1"),
            ("p0", "Concentração Inicial do Produto (mol/L)", "0"),
            ("t_input", "Tempo de Simulação (min)", "60"),
            ("Km", "Constante Michaelis (Km)", "1.0")
        ]
        for key, label, default in mm_parametros:
            row = QHBoxLayout()
            lbl = QLabel(label)
            input_field = QLineEdit()
            input_field.setText(default)
            self.inputs_mm[key] = input_field
            row.addWidget(lbl)
            row.addWidget(input_field)
            mm_input_layout.addLayout(row)
        self.mm_input_group.setLayout(mm_input_layout)
        layout.addWidget(self.mm_input_group)

        # Botões de ação para simulação
        button_layout = QHBoxLayout()
        sim_button = QPushButton("Gerar Simulação")
        sim_button.clicked.connect(self.run_simulation)
        back_button = QPushButton("Voltar")
        back_button.clicked.connect(lambda: self.central_widget.setCurrentIndex(0))
        button_layout.addWidget(sim_button)
        button_layout.addWidget(back_button)
        layout.addLayout(button_layout)

        sim_widget.setLayout(layout)
        self.central_widget.addWidget(sim_widget)

    def modeling_ui(self):
        """Tela de Modelagem"""
        model_widget = QWidget()
        layout = QVBoxLayout()

        header = QLabel("Modelagem de Reações")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #00B0FF;")
        layout.addWidget(header)

        # Passo 1: Seleção de arquivo
        file_group = QGroupBox("1. Inserir Dados do Arquivo Excel")
        file_layout = QVBoxLayout()
        self.file_label = QLabel("Nenhum arquivo selecionado.")
        file_button = QPushButton("Selecionar Arquivo Excel")
        file_button.clicked.connect(self.load_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(file_button)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Passo 2 (Ping Pong): Entradas com rótulos
        self.conc_group = QGroupBox("2. Inserir Enzima e Substratos Iniciais (Ping Pong Bi-Bi)")
        conc_layout = QVBoxLayout()
        self.e0_input = QLineEdit("0.01")
        self.s0_a_input = QLineEdit("0.1")
        self.s0_b_input = QLineEdit("0.1")

        for label_text, widget in [
            ("[E]0 (mol/L)", self.e0_input),
            ("[S1]0 (mol/L)", self.s0_a_input),
            ("[S2]0 (mol/L)", self.s0_b_input),
        ]:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(120)
            row.addWidget(label)
            row.addWidget(widget)
            conc_layout.addLayout(row)
        self.conc_group.setLayout(conc_layout)

        # Passo 3: Parâmetros da função
        param_group = QGroupBox("3. Inserir Parâmetros da Função")
        param_layout = QVBoxLayout()
        self.maxiter_input = QLineEdit("10")
        self.popsize_input = QLineEdit("5")
        self.mutation_input = QLineEdit("0.75,1")
        self.recombination_input = QLineEdit("0.8")

        for label_text, widget in [
            ("maxiter", self.maxiter_input),
            ("popsize", self.popsize_input),
            ("mutation", self.mutation_input),
            ("recombination", self.recombination_input),
        ]:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(120)
            row.addWidget(label)
            row.addWidget(widget)
            param_layout.addLayout(row)
        param_group.setLayout(param_layout)

        # Layout horizontal para passos 2 e 3
        pp_horizontal_group = QHBoxLayout()
        pp_horizontal_group.addWidget(self.conc_group)
        pp_horizontal_group.addWidget(param_group)
        layout.addLayout(pp_horizontal_group)

        # Passo 2 (Michaelis-Menten)
        self.mm_conc_group = QGroupBox("2. Inserir Parâmetros Michaelis Menten")
        mm_conc_layout = QVBoxLayout()
        self.e0_input_mm = QLineEdit("0.01")
        self.s0_input_mm = QLineEdit("0.1")

        for label_text, widget in [
            ("[E]0 (mol/L)", self.e0_input_mm),
            ("[S]0 (mol/L)", self.s0_input_mm),
        ]:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(120)
            row.addWidget(label)
            row.addWidget(widget)
            mm_conc_layout.addLayout(row)
        self.mm_conc_group.setLayout(mm_conc_layout)
        layout.addWidget(self.mm_conc_group)

        # Passo 4: Checkboxes para Ping Pong
        self.check_group_pingpong = QGroupBox("4. Ajustes Disponíveis (Ping Pong Bi-Bi)")
        check_layout_pingpong = QVBoxLayout()
        self.checkboxes_pingpong = {}
        for param in ["S1_adjust", "S2_adjust", "P1_adjust", "P2_adjust"]:
            self.checkboxes_pingpong[param] = QCheckBox(param)
            check_layout_pingpong.addWidget(self.checkboxes_pingpong[param])
        self.check_group_pingpong.setLayout(check_layout_pingpong)
        layout.addWidget(self.check_group_pingpong)

        # Checkboxes para Michaelis-Menten
        self.check_group_mm = QGroupBox("4. Ajustes Disponíveis (Michaelis Menten)")
        check_layout_mm = QVBoxLayout()
        self.checkboxes_mm = {}
        for param in ["S_adjust", "P_adjust"]:
            self.checkboxes_mm[param] = QCheckBox(param)
            check_layout_mm.addWidget(self.checkboxes_mm[param])
        self.check_group_mm.setLayout(check_layout_mm)
        layout.addWidget(self.check_group_mm)

        # Botões
        run_button = QPushButton("Gerar Modelagem")
        run_button.clicked.connect(self.run_modeling)
        back_button = QPushButton("Voltar")
        back_button.clicked.connect(lambda: self.central_widget.setCurrentIndex(0))
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(run_button)
        btn_layout.addWidget(back_button)
        layout.addLayout(btn_layout)

        # Saída textual dos parâmetros
        output_group = QGroupBox("Parâmetros Cinéticos")
        output_inner_layout = QVBoxLayout()
        self.parametros_output = QTextEdit()
        self.parametros_output.setReadOnly(True)
        self.parametros_output.setPlaceholderText("Os parâmetros cinéticos aparecerão aqui...")
        output_inner_layout.addWidget(self.parametros_output)
        output_group.setLayout(output_inner_layout)
        layout.addWidget(output_group)
        model_widget.setLayout(layout)
        self.central_widget.addWidget(model_widget)


    def run_simulation(self):
        if self.method_michaelis.isChecked():
            try:
                E0 = float(self.inputs_mm["E0"].text())
                Kcat = float(self.inputs_mm["Kcat"].text())
                s0 = float(self.inputs_mm["s0"].text())
                p0 = float(self.inputs_mm["p0"].text())
                t_input = float(self.inputs_mm["t_input"].text())
                Km = float(self.inputs_mm["Km"].text())
                funcoes_simulacao_mm.plot_sim(E0, Kcat, s0, p0, t_input, Km)
            except Exception as e:
                QMessageBox.critical(self, "Erro na Simulação Michaelis Menten", f"Erro: {str(e)}")
            return
        else:
            try:
                params = [float(self.inputs[key].text()) for key in self.inputs]
                funcoes_simulacao.plot_sim(*params)
            except Exception as e:
                QMessageBox.critical(self, "Erro na Simulação", f"Erro: {str(e)}")

    def run_modeling(self) -> None:
        # ───── verificação do arquivo ──────────────────────────────────────────
        if not hasattr(self, "excel_file"):
            QMessageBox.critical(self, "Erro", "Selecione um arquivo Excel primeiro.")
            return

        # ───── parâmetros comuns da Differential Evolution ────────────────────
        try:
            maxiter = int(self.maxiter_input.text()) if self.maxiter_input.text() else 1000
            popsize = int(self.popsize_input.text()) if self.popsize_input.text() else 50
            mutation = (0.5, 1.0)                              # default
            if self.mutation_input.text():
                mutation = tuple(float(x.strip()) for x in self.mutation_input.text().split(","))
            recombination = float(self.recombination_input.text()) if self.recombination_input.text() else 0.7
        except ValueError:
            QMessageBox.critical(self, "Erro", "Parâmetros de DE inválidos.")
            return

        # ======================================================================
        #  MICHAELIS–MENTEN
        # ======================================================================
        if self.method_michaelis.isChecked():
            try:
                from funcao_modelagem_michaelis import fit_mm                       # refatorado
            except ImportError as exc:
                QMessageBox.critical(self, "Erro", f"Falha ao importar mm_model.py:\n{exc}")
                return

            try:
                params, rmse_map, r2_map, fig = fit_mm(
                    self.excel_file,
                    maxiter=maxiter,
                    popsize=popsize,
                    mutation=mutation,
                    recombination=recombination
                )
            except Exception as exc:
                QMessageBox.critical(self, "Erro", f"Falha na modelagem Michaelis–Menten:\n{exc}")
                return

            # ── texto de R² ───────────────────────────────────────────────────
            r2_text = "───  R² das variáveis ajustadas  ───\n"
            for var, r2 in r2_map.items():
                r2_text += f"{var}: {r2:8.4f}\n"
            r2_text += "──────────────────────────────────────\n"

            # ── texto de parâmetros ───────────────────────────────────────────
            param_text = (
                f"Vmax : {params['Vmax']:.6e}\n"
                f"Km   : {params['Km']:.6e}"
            )
            self.parametros_output.setText(r2_text + param_text)

            # ── exibe figura ─────────────────────────────────────────────────
            canvas = FigureCanvas(fig)
            canvas.setWindowTitle("Resultados — Michaelis–Menten")
            canvas.show()
            return   # <<< encerra aqui para MM

        # ======================================================================
        #  PING–PONG Bi‑Bi
        # ======================================================================
        try:
            E0_guess = float(self.e0_input.text())
            s0_a     = float(self.s0_a_input.text())
            s0_b     = float(self.s0_b_input.text())
        except ValueError:
            QMessageBox.critical(self, "Erro", "Valores iniciais inválidos para E₀, S1 ou S2.")
            return

        adj = {k: cb.isChecked() for k, cb in self.checkboxes_pingpong.items()}

        try:
            from funcoes_modelagem_final import funcao_final
        except ImportError as exc:
            QMessageBox.critical(self, "Erro", f"Falha ao importar funcoes_modelagem.py:\n{exc}")
            return

        try:
            params, plot_results, r2_map = funcao_final(
                self.excel_file,
                E0_guess, s0_a, s0_b, adj,
                maxiter=maxiter, popsize=popsize,
                mutation=mutation, recombination=recombination
            )
        except Exception as exc:
            QMessageBox.critical(self, "Erro", f"Falha na modelagem Ping‑Pong:\n{exc}")
            return

        # ── texto de R² ───────────────────────────────────────────────────────
        r2_text = "───  R² das variáveis ajustadas  ───\n"
        for key, (lbl, r2) in r2_map.items():
            if adj.get(key, False):
                r2_text += f"{lbl}: {r2:8.4f}\n"
        r2_text += "──────────────────────────────────────\n"

        # ── desempacota parâmetros (10 valores) ───────────────────────────────
        (e0_opt, k1, k_1, k2, k3, k_3, k4,
        Km_A, Km_B, Vmax) = params

        k_app      = min(k2, k4)
        ratio_p1p2 = (k2 / k4) if k4 else float('inf')

        param_text = (
            f"E0    : {e0_opt:.6f}\n"
            f"k1    : {k1:.6f}\n"
            f"k_1   : {k_1:.6f}\n"
            f"k2    : {k2:.6f}\n"
            f"k3    : {k3:.6f}\n"
            f"k_3   : {k_3:.6f}\n"
            f"k4    : {k4:.6f}\n"
            f"Km_A  : {Km_A:.6f}\n"
            f"Km_B  : {Km_B:.6f}\n"
            f"Vmax  : {Vmax:.6f}\n"
            f"k_app : {k_app:.6f}\n"
            f"k2/k4 : {ratio_p1p2:.6f}"
        )

        self.parametros_output.setText(r2_text + param_text)

        # ── plota resultados Ping‑Pong ────────────────────────────────────────
        fig = plot_results()
        canvas = FigureCanvas(fig)
        canvas.setWindowTitle("Resultados — Ping‑Pong Bi‑Bi")
        canvas.show()


    def load_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Selecione o arquivo Excel", "", "Excel Files (*.xlsx; *.xls)", options=options
        )
        if file_name:
            self.file_label.setText(f"Arquivo: {os.path.basename(file_name)}")
            self.excel_file = file_name

            import pandas as pd
            try:
                data = pd.read_excel(file_name, dtype=str)
                data = data.applymap(lambda x: str(x).replace(',', '.')).astype(float)

                # Inicializa entradas com dados da primeira linha
                if 'Substrato 1' in data.columns:
                    self.s0_a_input.setText(str(data['Substrato 1'].iloc[0]))
                    self.s0_input_mm.setText(str(data['Substrato 1'].iloc[0]))
                if 'Substrato 2' in data.columns:
                    self.s0_b_input.setText(str(data['Substrato 2'].iloc[0]))

                self.excel_data = data

            except Exception as e:
                QMessageBox.warning(self, "Erro na leitura do arquivo", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
