import flet as ft
import numpy as np
from scipy.stats import mode, linregress, binom, poisson, f_oneway
from collections import Counter
import math


# --- Funções de Cálculo Estatístico ---

def calcular_estatisticas_descritivas(data_str):
    """
    Calcula e retorna estatísticas descritivas para um conjunto de dados.
    """
    try:
        data = [float(x.strip()) for x in data_str.split(',') if x.strip()]
        if not data:
            return "Erro: Insira dados numéricos válidos separados por vírgula."

        n = len(data)
        media = np.mean(data)
        mediana = np.median(data)

        # Moda (pode haver múltiplas)
        contagem = Counter(data)
        max_frequencia = 0
        modas = []
        for valor, freq in contagem.items():
            if freq > max_frequencia:
                max_frequencia = freq
                modas = [valor]
            elif freq == max_frequencia and max_frequencia > 1:
                modas.append(valor)
        if not modas and n > 0:
            modo_str = "Não há moda (todos os elementos são únicos)"
        elif modas and max_frequencia == 1:
            modo_str = "Não há moda (todos os elementos são únicos)"
        else:
            modo_str = ", ".join(map(str, sorted(modas)))

        amplitude = np.max(data) - np.min(data)
        variancia = np.var(data)
        desvio_padrao = np.std(data)

        results = (
            f"Resultados da Análise Descritiva:\n"
            f"  Número de Dados (n): {n}\n"
            f"  Média: {media:.4f}\n"
            f"  Mediana: {mediana:.4f}\n"
            f"  Moda: {modo_str}\n"
            f"  Amplitude: {amplitude:.4f}\n"
            f"  Variância (Pop.): {variancia:.4f}\n"
            f"  Desvio Padrão (Pop.): {desvio_padrao:.4f}"
        )
        return results
    except ValueError:
        return "Erro: Verifique se os dados são numéricos válidos separados por vírgula."
    except Exception as e:
        return f"Ocorreu um erro: {e}"


def calcular_regressao_linear(x_str, y_str):
    """
    Calcula e retorna os resultados da regressão linear simples.
    """
    try:
        x_data = [float(val.strip()) for val in x_str.split(',') if val.strip()]
        y_data = [float(val.strip()) for val in y_str.split(',') if val.strip()]

        if not x_data or not y_data or len(x_data) != len(y_data):
            return "Erro: Insira listas de dados numéricos X e Y válidas e de mesmo tamanho."

        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        r_squared = r_value ** 2

        results = (
            f"Resultados da Regressão Linear Simples:\n"
            f"  Coeficiente Angular (b): {slope:.4f}\n"
            f"  Intercepto (a): {intercept:.4f}\n"
            f"  Coeficiente de Determinação (R²): {r_squared:.4f}\n"
            f"  Valor-p: {p_value:.4f}\n"
            f"  Erro Padrão do Coeficiente: {std_err:.4f}"
        )
        return results
    except ValueError:
        return "Erro: Verifique se os dados são numéricos válidos separados por vírgula."
    except Exception as e:
        return f"Ocorreu um erro: {e}"


def calcular_teorema_bayes(pa_str, pb_dado_a_str, pb_dado_nao_a_str):
    """
    Calcula e retorna a probabilidade condicional P(A|B) usando o Teorema de Bayes,
    derivando P(B) através da Lei da Probabilidade Total.
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
        # P(B) = P(B|A) * P(A) + P(B|não A) * P(não A)
        PB = (PB_dado_A * PA) + (PB_dado_NAO_A * P_NAO_A)

        if PB == 0:
            return "Erro: O cálculo de P(B) resultou em zero, divisão por zero impossível."

        # Calcula P(A|B)
        # P(A|B) = (P(B|A) * P(A)) / P(B)
        PA_dado_B = (PB_dado_A * PA) / PB

        results = (
            f"Resultados do Teorema de Bayes:\n"
            f"  P(A) = {PA:.4f}\n"
            f"  P(B|A) = {PB_dado_A:.4f}\n"
            f"  P(B|não A) = {PB_dado_NAO_A:.4f}\n"
            f"  --------------------------\n"
            f"  P(não A) = 1 - P(A) = {P_NAO_A:.4f}\n"
            f"  P(B) = P(B|A)P(A) + P(B|não A)P(não A) = {PB:.4f}\n"
            f"  P(A|B) = (P(B|A) * P(A)) / P(B)\n"
            f"  P(A|B) = ({PB_dado_A:.4f} * {PA:.4f}) / {PB:.4f}\n"
            f"  P(A|B) = {PA_dado_B:.4f}"
        )
        return results
    except ValueError:
        return "Erro: Insira valores numéricos válidos para as probabilidades."
    except Exception as e:
        return f"Ocorreu um erro: {e}"


def calcular_distribuicao_binomial(n_str, p_str, k_str):
    """
    Calcula e retorna a probabilidade para a Distribuição Binomial.
    """
    try:
        n = int(n_str)
        p = float(p_str)
        k = int(k_str)

        if n <= 0 or not (0 <= p <= 1) or k < 0 or k > n:
            return "Erro: Verifique os valores. n > 0, 0 <= p <= 1, 0 <= k <= n."

        prob = binom.pmf(k, n, p)

        results = (
            f"Resultados da Distribuição Binomial:\n"
            f"  Número de Tentativas (n): {n}\n"
            f"  Probabilidade de Sucesso (p): {p:.4f}\n"
            f"  Número de Sucessos (k): {k}\n"
            f"  P(X = {k}) = {prob:.6f}"
        )
        return results
    except ValueError:
        return "Erro: Insira valores numéricos e inteiros válidos para n e k, e numérico para p."
    except Exception as e:
        return f"Ocorreu um erro: {e}"


def calcular_distribuicao_poisson(lam_str, k_str):
    """
    Calcula e retorna a probabilidade para a Distribuição de Poisson.
    """
    try:
        lam = float(lam_str)
        k = int(k_str)

        if lam <= 0 or k < 0:
            return "Erro: Lambda deve ser > 0 e k deve ser >= 0."

        prob = poisson.pmf(k, lam)

        results = (
            f"Resultados da Distribuição de Poisson:\n"
            f"  Taxa Média (λ): {lam:.4f}\n"
            f"  Número de Eventos (k): {k}\n"
            f"  P(X = {k}) = {prob:.6f}"
        )
        return results
    except ValueError:
        return "Erro: Insira valores numéricos válidos para Lambda e k (inteiro)."
    except Exception as e:
        return f"Ocorreu um erro: {e}"


def calcular_anova(groups_str):
    """
    Realiza a Análise de Variância (ANOVA) para múltiplos grupos de dados.
    """
    try:
        groups_data = []
        for group_str in groups_str.split(';'):
            data = [float(x.strip()) for x in group_str.split(',') if x.strip()]
            if data:
                groups_data.append(data)

        if len(groups_data) < 2:
            return "Erro: ANOVA requer pelo menos dois grupos de dados separados por ';'."

        # Realiza a ANOVA usando f_oneway
        F_statistic, p_value = f_oneway(*groups_data)

        results = (
            f"Resultados da Análise de Variância (ANOVA):\n"
            f"  Número de Grupos Analisados: {len(groups_data)}\n"
            f"  Estatística F: {F_statistic:.4f}\n"
            f"  Valor-p: {p_value:.4f}\n"
            f"  Interpretação (regra geral, α=0.05):\n"
            f"    - Se p-valor < 0.05: Há diferença estatisticamente significativa entre as médias dos grupos.\n"
            f"    - Se p-valor >= 0.05: Não há diferença estatisticamente significativa entre as médias dos grupos."
        )
        return results
    except ValueError:
        return "Erro: Verifique o formato dos dados. Use vírgulas para separar valores dentro de um grupo e ponto e vírgula para separar grupos (ex: 1,2,3;4,5,6)."
    except Exception as e:
        return f"Ocorreu um erro: {e}"


# --- Componentes da UI Flet ---

def main(page: ft.Page):
    page.title = "Calculadora Estatística e Probabilística"
    page.vertical_alignment = ft.CrossAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_width = 800
    page.window_height = 700
    page.scroll = ft.ScrollMode.ADAPTIVE

    # Campo de resultados
    results_text = ft.Container(
        content=ft.Text(value="Selecione uma função para começar...", selectable=True),
        padding=10,
        margin=ft.margin.only(top=20),
        bgcolor=ft.colors.BLUE_GREY_100,
        border_radius=ft.border_radius.all(5),
        width=600,
        height=200,
        alignment=ft.alignment.top_left
    )

    # Área de inputs dinâmicos
    input_area = ft.Column(
        spacing=10,
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        width=600
    )

    # --- Funções para alternar as interfaces e calcular ---

    # --- Análise Descritiva ---
    data_input_desc = ft.TextField(
        label="Dados (separe por vírgulas, ex: 1.2, 3.4, 5)",
        hint_text="Ex: 10, 15, 20, 25, 30",
        multiline=True,
        min_lines=3,
        max_lines=5,
        width=500,
        border_radius=ft.border_radius.all(8)
    )

    def on_descriptive_stats_calculate(e):
        results_text.content.value = calcular_estatisticas_descritivas(data_input_desc.value)
        page.update()

    def show_descriptive_stats_inputs():
        input_area.controls.clear()
        input_area.controls.extend([data_input_desc,
                                    ft.ElevatedButton("Calcular Estatísticas Descritivas",
                                                      on_click=on_descriptive_stats_calculate)])
        page.update()

    # --- Regressão Linear ---
    x_input_reg = ft.TextField(
        label="Dados de X (separe por vírgulas, ex: 1, 2, 3)",
        hint_text="Ex: 10, 20, 30, 40",
        width=500,
        border_radius=ft.border_radius.all(8)
    )
    y_input_reg = ft.TextField(
        label="Dados de Y (separe por vírgulas, ex: 10, 25, 35)",
        hint_text="Ex: 12, 28, 33, 45",
        width=500,
        border_radius=ft.border_radius.all(8)
    )

    def on_linear_regression_calculate(e):
        results_text.content.value = calcular_regressao_linear(x_input_reg.value, y_input_reg.value)
        page.update()

    def show_linear_regression_inputs():
        input_area.controls.clear()
        input_area.controls.extend([x_input_reg, y_input_reg,
                                    ft.ElevatedButton("Calcular Regressão Linear",
                                                      on_click=on_linear_regression_calculate)])
        page.update()

    # --- Teorema de Bayes ---
    pa_input_bayes = ft.TextField(label="P(A) - Probabilidade de A", hint_text="Ex: 0.5", width=500,
                                  border_radius=ft.border_radius.all(8))
    pb_dado_a_input_bayes = ft.TextField(label="P(B|A) - Probabilidade de B dado A", hint_text="Ex: 0.8", width=500,
                                         border_radius=ft.border_radius.all(8))
    pb_dado_nao_a_input_bayes = ft.TextField(label="P(B|não A) - Probabilidade de B dado não A", hint_text="Ex: 0.2",
                                             width=500, border_radius=ft.border_radius.all(8))  # Novo campo

    def on_bayes_theorem_calculate(e):
        results_text.content.value = calcular_teorema_bayes(
            pa_input_bayes.value,
            pb_dado_a_input_bayes.value,
            pb_dado_nao_a_input_bayes.value
        )
        page.update()

    def show_bayes_theorem_inputs():
        input_area.controls.clear()
        input_area.controls.extend([
            pa_input_bayes,
            pb_dado_a_input_bayes,
            pb_dado_nao_a_input_bayes,
            ft.ElevatedButton("Calcular Teorema de Bayes", on_click=on_bayes_theorem_calculate)
        ])
        page.update()

    # --- Distribuição Binomial ---
    n_input_binom = ft.TextField(label="n - Número de tentativas", hint_text="Ex: 10", width=500,
                                 border_radius=ft.border_radius.all(8))
    p_input_binom = ft.TextField(label="p - Probabilidade de sucesso (0 a 1)", hint_text="Ex: 0.5", width=500,
                                 border_radius=ft.border_radius.all(8))
    k_input_binom = ft.TextField(label="k - Número de sucessos desejados", hint_text="Ex: 3", width=500,
                                 border_radius=ft.border_radius.all(8))

    def on_binomial_distribution_calculate(e):
        results_text.content.value = calcular_distribuicao_binomial(n_input_binom.value, p_input_binom.value,
                                                                    k_input_binom.value)
        page.update()

    def show_binomial_distribution_inputs():
        input_area.controls.clear()
        input_area.controls.extend([n_input_binom, p_input_binom, k_input_binom,
                                    ft.ElevatedButton("Calcular Distribuição Binomial",
                                                      on_click=on_binomial_distribution_calculate)])
        page.update()

    # --- Distribuição de Poisson ---
    lambda_input_poisson = ft.TextField(label="λ (Lambda) - Taxa média de ocorrências", hint_text="Ex: 3.5", width=500,
                                        border_radius=ft.border_radius.all(8))
    k_input_poisson = ft.TextField(label="k - Número de eventos desejados", hint_text="Ex: 2", width=500,
                                   border_radius=ft.border_radius.all(8))

    def on_poisson_distribution_calculate(e):
        results_text.content.value = calcular_distribuicao_poisson(lambda_input_poisson.value, k_input_poisson.value)
        page.update()

    def show_poisson_distribution_inputs():
        input_area.controls.clear()
        input_area.controls.extend([lambda_input_poisson, k_input_poisson,
                                    ft.ElevatedButton("Calcular Distribuição de Poisson",
                                                      on_click=on_poisson_distribution_calculate)])
        page.update()

    # --- ANOVA ---
    groups_input_anova = ft.TextField(
        label="Grupos de Dados (separe valores por vírgula, grupos por ponto e vírgula)",
        hint_text="Ex: 10,12,11; 15,14,16; 9,10,8",
        multiline=True,
        min_lines=3,
        max_lines=5,
        width=500,
        border_radius=ft.border_radius.all(8)
    )

    def on_anova_calculate(e):
        results_text.content.value = calcular_anova(groups_input_anova.value)
        page.update()

    def show_anova_inputs():
        input_area.controls.clear()
        input_area.controls.extend([groups_input_anova,
                                    ft.ElevatedButton("Calcular ANOVA", on_click=on_anova_calculate)])
        page.update()

    # --- Menu de Botões das Funções ---
    menu_buttons = ft.Row(
        [
            ft.ElevatedButton("Análise Descritiva", on_click=lambda e: show_descriptive_stats_inputs(),
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))),
            ft.ElevatedButton("Regressão Linear", on_click=lambda e: show_linear_regression_inputs(),
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))),
            ft.ElevatedButton("Teorema de Bayes", on_click=lambda e: show_bayes_theorem_inputs(),
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))),
            ft.ElevatedButton("Binomial", on_click=lambda e: show_binomial_distribution_inputs(),
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))),
            ft.ElevatedButton("Poisson", on_click=lambda e: show_poisson_distribution_inputs(),
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))),
            ft.ElevatedButton("ANOVA", on_click=lambda e: show_anova_inputs(),
                              style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10))),
        ],
        wrap=True,
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=10
    )

    # Layout da página
    page.add(
        ft.Column(
            [
                ft.Text("Calculadora Estatística e Probabilística", size=28, weight=ft.FontWeight.BOLD),
                ft.Divider(height=20, color=ft.colors.BLUE_GREY_200),
                menu_buttons,
                ft.Divider(height=20, color=ft.colors.BLUE_GREY_200),
                input_area,
                results_text,
            ],
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
            width=800
        )
    )


# --- Iniciar o Aplicativo Flet ---
if __name__ == "__main__":
    ft.app(target=main)
