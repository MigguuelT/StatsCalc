# 🚀 Calculadora Estatística e Probabilística 🚀

Bem-vindo à **Calculadora Estatística e Probabilística**! Este repositório é o seu centro de comando para realizar cálculos estatísticos e probabilísticos complexos de forma rápida, precisa e intuitiva. Se você é um estudante, um profissional de dados, ou simplesmente um curioso sobre o mundo dos números, estas ferramentas foram feitas para você!

## 📊 A Calculadora Estatística Principal

Nossa principal ferramenta é uma poderosa Calculadora Estatística que vem em duas versões, ambas com um conjunto robusto de funcionalidades para atender às suas necessidades de análise de dados.

### Interfaces Disponíveis

1.  **Versão Flet (Recomendada):** Uma interface moderna, limpa e rápida, construída com o framework Flet. Perfeita para quem busca agilidade e uma experiência de usuário fluida.
2.  **Versão PySide (com Gráficos):** Uma aplicação de desktop clássica, construída com PySide6 (Qt). O grande diferencial desta versão é a **visualização gráfica** para a análise de Regressão Linear, permitindo que você veja a linha de regressão em relação aos seus dados.

### Funcionalidades

Ambas as versões da calculadora incluem as seguintes análises:

*   **Análise Descritiva:** Calcule rapidamente as métricas fundamentais de qualquer conjunto de dados:
    *   Média, Mediana e Moda
    *   Variância e Desvio Padrão
    *   Amplitude (valor máximo - mínimo)
*   **Regressão Linear Simples:** Encontre a linha de melhor ajuste para seus dados bivariados.
    *   Coeficiente Angular (b) e Intercepto (a)
    *   Coeficiente de Determinação (R²) para avaliar a qualidade do ajuste.
    *   A versão PySide plota um **gráfico de dispersão** com a reta de regressão!
*   **Teorema de Bayes:** Calcule a probabilidade condicional de um evento com base em conhecimentos prévios. Ideal para problemas de diagnóstico e inferência.
*   **Distribuição Binomial:** Modele o número de sucessos em uma sequência de *n* tentativas independentes.
*   **Distribuição de Poisson:** Analise a probabilidade de um número de eventos ocorrer em um intervalo fixo de tempo ou espaço.
*   **Análise de Variância (ANOVA):** Compare as médias de dois ou mais grupos e determine se há diferenças estatisticamente significativas entre eles.

## 🔢 Bônus: Calculadora Simples

Além da suíte estatística, incluímos uma **Calculadora Simples** para operações aritméticas do dia a dia. Rápida, leve e com um design elegante, é perfeita para quando você precisa de um cálculo rápido sem sair do seu ambiente de desenvolvimento.

## ⚙️ Como Usar

Para começar a usar as calculadoras, siga os passos abaixo.

### 1. Pré-requisitos

Certifique-se de que você tem o **Python 3** instalado em seu sistema.

### 2. Instalação das Dependências

Abra o seu terminal ou prompt de comando na pasta deste repositório e instale todas as bibliotecas necessárias com um único comando:

```bash
pip install -r requirements.txt
```

### 3. Executando as Calculadoras

Com tudo instalado, você pode iniciar a calculadora que desejar:

*   **Para a Calculadora Estatística (Flet - Recomendado):**
    ```bash
    python StatsCalc_2.py
    ```

*   **Para a Calculadora Estatística (PySide - com Gráficos):**
    ```bash
    python stats_calc_pyside.py
    ```

*   **Para a Calculadora Simples:**
    ```bash
    python calc_simples.py
    ```

## ✨ Resumo das Features

| Feature                 | Calculadora Estatística (Flet) | Calculadora Estatística (PySide) | Calculadora Simples |
| ----------------------- | :----------------------------: | :------------------------------: | :-----------------: |
| Análise Descritiva      |               ✅               |                ✅                |          ❌         |
| Regressão Linear        |               ✅               |                ✅                |          ❌         |
| Teorema de Bayes        |               ✅               |                ✅                |          ❌         |
| Distribuição Binomial   |               ✅               |                ✅                |          ❌         |
| Distribuição de Poisson |               ✅               |                ✅                |          ❌         |
| ANOVA                   |               ✅               |                ✅                |          ❌         |
| **Gráfico de Regressão**  |               ❌               |                ✅                |          ❌         |
| Operações Aritméticas   |               ❌               |                ❌                |          ✅         |
| Interface Gráfica       |             Moderna            |             Clássica             |       Simples       |

## 🤝 Contribuições

Sinta-se à vontade para abrir *issues* com sugestões, reportar bugs ou submeter *pull requests*. Sua contribuição é muito bem-vinda!
