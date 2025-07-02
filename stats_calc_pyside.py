import sys
import numpy as np
from scipy.stats import linregress, binom, poisson, f_oneway
from collections import Counter
import math

import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QStackedWidget, QFrame,
    QSplitter  # Importar o QSplitter
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


# ===================================================================
# PARTE 1: FUNÇÕES DE CÁLCULO (Sem alterações)
# ===================================================================
# ... (todo o bloco de funções de cálculo permanece o mesmo) ...
def calcular_estatisticas_descritivas(data_str):
    try:
        data = [float(x.strip()) for x in data_str.split(',') if x.strip()]
        if not data: return "Erro: Insira dados numéricos válidos separados por vírgula."
        n, media, mediana = len(data), np.mean(data), np.median(data)
        contagem = Counter(data)
        max_frequencia = 0
        modas = []
        for valor, freq in contagem.items():
            if freq > max_frequencia:
                max_frequencia, modas = freq, [valor]
            elif freq == max_frequencia and max_frequencia > 1:
                modas.append(valor)
        if not modas or max_frequencia == 1:
            modo_str = "Amodal (sem repetições significativas)"
        else:
            modo_str = ", ".join(map(str, sorted(modas)))
        amplitude, variancia, desvio_padrao = np.max(data) - np.min(data), np.var(data), np.std(data)
        return (f"Resultados da Análise Descritiva:\n"
                f"  Número de Dados (n): {n}\n"
                f"  Média: {media:.4f}\n"
                f"  Mediana: {mediana:.4f}\n"
                f"  Moda: {modo_str}\n"
                f"  Amplitude: {amplitude:.4f}\n"
                f"  Variância (Pop.): {variancia:.4f}\n"
                f"  Desvio Padrão (Pop.): {desvio_padrao:.4f}")
    except (ValueError, TypeError):
        return "Erro: Verifique se os dados são numéricos válidos."
    except Exception as e:
        return f"Ocorreu um erro: {e}"


def calcular_regressao_linear(x_str, y_str):
    try:
        x_data = [float(val.strip()) for val in x_str.split(',') if val.strip()]
        y_data = [float(val.strip()) for val in y_str.split(',') if val.strip()]
        if not x_data or len(x_data) != len(y_data):
            return {"error": "Erro: Insira listas X e Y válidas e de mesmo tamanho."}
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        text_result = (
            f"Resultados da Regressão Linear Simples:\n"
            f"  Equação: Y = {intercept:.4f} + {slope:.4f}X\n"
            f"  Coeficiente Angular (b): {slope:.4f}\n"
            f"  Intercepto (a): {intercept:.4f}\n"
            f"  Coeficiente de Determinação (R²): {r_value ** 2:.4f}\n"
            f"  Valor-p: {p_value:.4f}"
        )
        plot_data = {
            'x_data': x_data,
            'y_data': y_data,
            'line_x': np.array(x_data),
            'line_y': intercept + slope * np.array(x_data)
        }
        return {"text": text_result, "plot_data": plot_data}
    except (ValueError, TypeError):
        return {"error": "Erro: Verifique se os dados são numéricos válidos."}
    except Exception as e:
        return {"error": f"Ocorreu um erro: {e}"}


def calcular_teorema_bayes(pa_str, pb_dado_a_str, pb_dado_nao_a_str):
    """
    Calcula e retorna a probabilidade P(A|B) detalhando os passos do cálculo.
    """
    try:
        PA = float(pa_str)
        PB_dado_A = float(pb_dado_a_str)
        PB_dado_NAO_A = float(pb_dado_nao_a_str)

        if not (0 <= PA <= 1 and 0 <= PB_dado_A <= 1 and 0 <= PB_dado_NAO_A <= 1):
            return "Erro: As probabilidades devem estar entre 0 e 1."

        # Calcula P(não A)
        P_NAO_A = 1 - PA

        # Calcula P(B) usando a Lei da Probabilidade Total
        PB = (PB_dado_A * PA) + (PB_dado_NAO_A * P_NAO_A)

        if PB == 0:
            return "Erro: P(B) resultou em zero, divisão por zero impossível."

        # Calcula P(A|B) usando o Teorema de Bayes
        PA_dado_B = (PB_dado_A * PA) / PB

        # Constrói a string de resultado detalhada
        results = (
            f"Resultados do Teorema de Bayes:\n\n"
            f"Entradas:\n"
            f"  P(A) = {PA:.4f}\n"
            f"  P(B|A) = {PB_dado_A:.4f}\n"
            f"  P(B|¬A) = {PB_dado_NAO_A:.4f}\n"
            f"  (Calculado) P(¬A) = 1 - {PA:.4f} = {P_NAO_A:.4f}\n"
            f"--------------------------------------------------\n"
            f"1. Cálculo de P(B) pela Lei da Probabilidade Total:\n"
            f"   Fórmula: P(B) = P(B|A) * P(A) + P(B|¬A) * P(¬A)\n"
            f"   Cálculo: P(B) = ({PB_dado_A:.4f} * {PA:.4f}) + ({PB_dado_NAO_A:.4f} * {P_NAO_A:.4f})\n"
            f"   Resultado: P(B) = {PB:.4f}\n"
            f"--------------------------------------------------\n"
            f"2. Cálculo de P(A|B) pelo Teorema de Bayes:\n"
            f"   Fórmula: P(A|B) = [P(B|A) * P(A)] / P(B)\n"
            f"   Cálculo: P(A|B) = [{PB_dado_A:.4f} * {PA:.4f}] / {PB:.4f}\n\n"
            f"   Resultado Final: P(A|B) = {PA_dado_B:.4f}"
        )
        return results

    except (ValueError, TypeError):
        return "Erro: Insira valores numéricos válidos para as probabilidades."
    except Exception as e:
        return f"Ocorreu um erro: {e}"


def calcular_distribuicao_binomial(n_str, p_str, k_str):
    try:
        n, p, k = int(n_str), float(p_str), int(k_str)
        if not (n > 0 and 0 <= p <= 1 and 0 <= k <= n):
            return "Erro: Verifique os valores (n>0, 0<=p<=1, 0<=k<=n)."
        prob = binom.pmf(k, n, p)
        return (f"Resultados da Distribuição Binomial:\n"
                f"  Número de Tentativas (n): {n}\n"
                f"  Probabilidade de Sucesso (p): {p:.4f}\n"
                f"  Número de Sucessos (k): {k}\n"
                f"  P(X = {k}) = {prob:.6f}")
    except (ValueError, TypeError):
        return "Erro: Insira valores válidos (n, k inteiros)."
    except Exception as e:
        return f"Ocorreu um erro: {e}"


def calcular_distribuicao_poisson(lam_str, k_str):
    try:
        lam, k = float(lam_str), int(k_str)
        if not (lam > 0 and k >= 0):
            return "Erro: Lambda (λ) deve ser > 0 e k deve ser >= 0."
        prob = poisson.pmf(k, lam)
        return (f"Resultados da Distribuição de Poisson:\n"
                f"  Taxa Média (λ): {lam:.4f}\n"
                f"  Número de Eventos (k): {k}\n"
                f"  P(X = {k}) = {prob:.6f}")
    except (ValueError, TypeError):
        return "Erro: Insira valores válidos (k inteiro)."
    except Exception as e:
        return f"Ocorreu um erro: {e}"


def calcular_anova(groups_str):
    try:
        groups_data = []
        for group_str in groups_str.split(';'):
            data = [float(x.strip()) for x in group_str.split(',') if x.strip()]
            if data: groups_data.append(data)
        if len(groups_data) < 2: return "Erro: ANOVA requer pelo menos dois grupos."
        F_statistic, p_value = f_oneway(*groups_data)
        return (f"Resultados da ANOVA:\n"
                f"  Número de Grupos: {len(groups_data)}\n"
                f"  Estatística F: {F_statistic:.4f}\n"
                f"  Valor-p: {p_value:.4f}\n\n"
                f"  Interpretação (α=0.05):\n"
                f"   - p < 0.05: Há diferença significativa entre as médias.\n"
                f"   - p >= 0.05: Não há diferença significativa entre as médias.")
    except (ValueError, TypeError):
        return "Erro: Verifique o formato dos dados."
    except Exception as e:
        return f"Ocorreu um erro: {e}"


# ===================================================================
# PARTE 2: CLASSE DA APLICAÇÃO PYQT
# ===================================================================

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class StatsCalcWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculadora Estatística e Probabilística")
        self.setGeometry(100, 100, 850, 750)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Calculadora Estatística e Probabilística")
        title_font = QFont();
        title_font.setPointSize(20);
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(title_label)

        self.main_layout.addWidget(self.create_divider())
        self.create_menu_buttons()
        self.main_layout.addWidget(self.create_divider())

        # --- INÍCIO DA MUDANÇA PARA QSplitter ---

        # 1. Criar o splitter principal com orientação vertical
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # 2. Criar a área de inputs (não muda)
        self.input_stack = self.create_input_stack()

        # 3. Criar um widget contêiner para a parte de baixo (resultados + gráfico)
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Os resultados aparecerão aqui...")
        # REMOVIDA a linha self.results_text.setFixedHeight(150)

        self.plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.plot_canvas.setVisible(False)

        bottom_layout.addWidget(self.results_text)
        bottom_layout.addWidget(self.plot_canvas)

        # 4. Adicionar a área de inputs e o contêiner de baixo ao splitter
        main_splitter.addWidget(self.input_stack)
        main_splitter.addWidget(bottom_widget)

        # 5. Ajustar o tamanho inicial das áreas no splitter (opcional, mas recomendado)
        # Dá 30% do espaço para a área de input e 70% para a de resultados/gráfico
        main_splitter.setSizes([150, 450])

        # 6. Adicionar o splitter principal ao layout da janela
        self.main_layout.addWidget(main_splitter)

        # --- FIM DA MUDANÇA PARA QSplitter ---

        self.btn_desc_stats.clicked.connect(lambda: self.show_page(0))
        self.btn_regressao.clicked.connect(lambda: self.show_page(1))
        self.btn_bayes.clicked.connect(lambda: self.show_page(2))
        self.btn_binomial.clicked.connect(lambda: self.show_page(3))
        self.btn_poisson.clicked.connect(lambda: self.show_page(4))
        self.btn_anova.clicked.connect(lambda: self.show_page(5))

        self.show_page(0)

    def show_page(self, index):
        self.input_stack.setCurrentIndex(index)
        self.plot_canvas.setVisible(False)
        self.results_text.clear()

    def create_divider(self):
        divider = QFrame();
        divider.setFrameShape(QFrame.Shape.HLine);
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        return divider

    def create_menu_buttons(self):
        menu_widget = QWidget()
        menu_layout = QHBoxLayout(menu_widget)
        menu_layout.setAlignment(Qt.AlignmentFlag.AlignCenter);
        menu_layout.setSpacing(10)
        self.btn_desc_stats = QPushButton("Análise Descritiva")
        self.btn_regressao = QPushButton("Regressão Linear")
        self.btn_bayes = QPushButton("Teorema de Bayes")
        self.btn_binomial = QPushButton("Binomial")
        self.btn_poisson = QPushButton("Poisson")
        self.btn_anova = QPushButton("ANOVA")
        buttons = [self.btn_desc_stats, self.btn_regressao, self.btn_bayes, self.btn_binomial, self.btn_poisson,
                   self.btn_anova]
        for btn in buttons:
            menu_layout.addWidget(btn)
        self.main_layout.addWidget(menu_widget)

    def create_input_stack(self):
        # Este método agora retorna o QStackedWidget criado
        input_stack = QStackedWidget()
        input_stack.addWidget(self.create_desc_stats_page())
        input_stack.addWidget(self.create_regression_page())
        input_stack.addWidget(self.create_bayes_page())
        input_stack.addWidget(self.create_binomial_page())
        input_stack.addWidget(self.create_poisson_page())
        input_stack.addWidget(self.create_anova_page())
        return input_stack

    # ... (O restante dos métodos para criar as páginas e rodar os cálculos permanece o mesmo) ...
    def create_page_layout(self, label_text):
        page = QWidget();
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop);
        layout.setSpacing(10)
        layout.addWidget(QLabel(label_text));
        return page, layout

    def create_desc_stats_page(self):
        page, layout = self.create_page_layout("<b>Análise Descritiva:</b> Insira os dados separados por vírgula.")
        desc_input = QTextEdit();
        desc_input.setPlaceholderText("Ex: 10, 15, 20, 25, 30")
        btn = QPushButton("Calcular Estatísticas")
        btn.clicked.connect(
            lambda: self.results_text.setText(calcular_estatisticas_descritivas(desc_input.toPlainText())))
        layout.addWidget(desc_input);
        layout.addWidget(btn);
        return page

    def create_regression_page(self):
        page, layout = self.create_page_layout("<b>Regressão Linear:</b> Insira os dados de X e Y.")
        self.reg_x_input = QLineEdit();
        self.reg_x_input.setPlaceholderText("Dados de X, separados por vírgula")
        self.reg_y_input = QLineEdit();
        self.reg_y_input.setPlaceholderText("Dados de Y, separados por vírgula")
        btn = QPushButton("Calcular Regressão e Gerar Gráfico")
        btn.clicked.connect(self.run_linear_regression)
        layout.addWidget(QLabel("Valores de X:"));
        layout.addWidget(self.reg_x_input)
        layout.addWidget(QLabel("Valores de Y:"));
        layout.addWidget(self.reg_y_input)
        layout.addWidget(btn);
        return page

    def run_linear_regression(self):
        result = calcular_regressao_linear(self.reg_x_input.text(), self.reg_y_input.text())
        if "error" in result:
            self.results_text.setText(result["error"])
            self.plot_canvas.setVisible(False)
        else:
            self.results_text.setText(result["text"])
            self.update_regression_plot(result["plot_data"])
            self.plot_canvas.setVisible(True)

    def update_regression_plot(self, plot_data):
        self.plot_canvas.axes.clear()
        self.plot_canvas.axes.scatter(plot_data['x_data'], plot_data['y_data'], label='Dados Originais')
        self.plot_canvas.axes.plot(plot_data['line_x'], plot_data['line_y'], color='red', linewidth=2,
                                   label='Reta de Regressão')
        self.plot_canvas.axes.set_title('Gráfico de Dispersão e Reta de Regressão')
        self.plot_canvas.axes.set_xlabel('Eixo X')
        self.plot_canvas.axes.set_ylabel('Eixo Y')
        self.plot_canvas.axes.legend()
        self.plot_canvas.axes.grid(True)
        self.plot_canvas.draw()

    def create_bayes_page(self):
        page, layout = self.create_page_layout("<b>Teorema de Bayes:</b> Insira as probabilidades (0 a 1).")
        bayes_pa_input = QLineEdit();
        bayes_pa_input.setPlaceholderText("Ex: 0.02")
        bayes_pba_input = QLineEdit();
        bayes_pba_input.setPlaceholderText("Ex: 0.85")
        bayes_pbna_input = QLineEdit();
        bayes_pbna_input.setPlaceholderText("Ex: 0.05")
        btn = QPushButton("Calcular P(A|B)")
        btn.clicked.connect(lambda: self.results_text.setText(
            calcular_teorema_bayes(bayes_pa_input.text(), bayes_pba_input.text(), bayes_pbna_input.text())))
        layout.addWidget(QLabel("Probabilidade de A  - P(A):"));
        layout.addWidget(bayes_pa_input)
        layout.addWidget(QLabel("Probabilidade de B dado A  - P(B|A):"));
        layout.addWidget(bayes_pba_input)
        layout.addWidget(QLabel("Probabilidade de B dado não A - P(B|¬A):"));
        layout.addWidget(bayes_pbna_input)
        layout.addWidget(btn);
        return page

    def create_binomial_page(self):
        page, layout = self.create_page_layout("<b>Distribuição Binomial:</b>")
        binom_n = QLineEdit();
        binom_n.setPlaceholderText("Número inteiro de tentativas")
        binom_p = QLineEdit();
        binom_p.setPlaceholderText("Probabilidade de sucesso (ex: 0.5)")
        binom_k = QLineEdit();
        binom_k.setPlaceholderText("Número inteiro de sucessos desejados")
        btn = QPushButton("Calcular Probabilidade Binomial")
        btn.clicked.connect(lambda: self.results_text.setText(
            calcular_distribuicao_binomial(binom_n.text(), binom_p.text(), binom_k.text())))
        layout.addWidget(QLabel("n (tentativas):"));
        layout.addWidget(binom_n)
        layout.addWidget(QLabel("p (probabilidade):"));
        layout.addWidget(binom_p)
        layout.addWidget(QLabel("k (sucessos):"));
        layout.addWidget(binom_k)
        layout.addWidget(btn);
        return page

    def create_poisson_page(self):
        page, layout = self.create_page_layout("<b>Distribuição de Poisson:</b>")
        poisson_lambda = QLineEdit();
        poisson_lambda.setPlaceholderText("Número médio de ocorrências")
        poisson_k = QLineEdit();
        poisson_k.setPlaceholderText("Número de eventos desejados")
        btn = QPushButton("Calcular Probabilidade de Poisson")
        btn.clicked.connect(
            lambda: self.results_text.setText(calcular_distribuicao_poisson(poisson_lambda.text(), poisson_k.text())))
        layout.addWidget(QLabel("Lambda (λ) - Taxa Média:"));
        layout.addWidget(poisson_lambda)
        layout.addWidget(QLabel("k - Número de Eventos:"));
        layout.addWidget(poisson_k)
        layout.addWidget(btn);
        return page

    def create_anova_page(self):
        page, layout = self.create_page_layout(
            "<b>ANOVA:</b> Insira grupos separados por ponto e vírgula (;) e valores por vírgula (,).")
        anova_input = QTextEdit();
        anova_input.setPlaceholderText("Ex: 10,12,11; 15,14,16; 9,10,8")
        btn = QPushButton("Calcular ANOVA")
        btn.clicked.connect(lambda: self.results_text.setText(calcular_anova(anova_input.toPlainText())))
        layout.addWidget(anova_input);
        layout.addWidget(btn);
        return page


# ===================================================================
# PARTE 3: INICIALIZAÇÃO DA APLICAÇÃO
# ===================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QWidget { font-size: 14px; }
        QPushButton {
            background-color: #E0E0E0; border: 1px solid #BDBDBD;
            padding: 8px 16px; border-radius: 5px;
        }
        QPushButton:hover { background-color: #DEDEDE; border: 1px solid #0078D7; }
        QPushButton:pressed { background-color: #C0C0C0; }
        QLineEdit, QTextEdit {
            border: 1px solid #BDBDBD; padding: 5px; border-radius: 5px;
        }
        QLineEdit:focus, QTextEdit:focus { border: 1px solid #0078D7; }
        QSplitter::handle { background-color: #E0E0E0; }
        QSplitter::handle:hover { background-color: #BDBDBD; }
        QSplitter::handle:pressed { background-color: #0078D7; }
    """)

    window = StatsCalcWindow()
    window.show()
    sys.exit(app.exec())
