Este projeto tem como objetivo minerar dados e gerar gráficos para a análise utilizando um dataset. O dataset escolhido se chama "Gold Price Regression" e está disponível para download através do link: https://www.kaggle.com/datasets/franciscogcc/financial-data. Aplcando técnicas de mineração de dados como visualizações, correlação entre variáveis e regressão linear e a linguagem de progamação Python serão gerados os gráficos. Para obter geração e interpretação dos gráficos siga os seguintes passos:

1. Configuração do Ambiente

Você precisará de:

  Python Instalado: Baixe e instale a versão mais recente de Python.

  Use o seguinte comando para a instalação das bibliotecas:
  
  pip install pandas matplotlib seaborn scikit-learn

2. Preparação dos Dados

    Obtenha o arquivo de dados (financial_regression.csv).
    Salve o arquivo no diretório de trabalho que você usará para rodar o script Python.

3. Execução do Código

    Abra um editor ou IDE de sua preferência:
        Jupyter Notebook (para visualizar gráficos diretamente).
        VS Code (com Python configurado).
        Terminal com qualquer editor de texto.

    Copie o código fornecido acima para um arquivo Python, por exemplo: gerar_graficos.py.

    Edite a linha referente ao caminho do arquivo para apontar corretamente para o arquivo de dados:

file_path = 'caminho/para/seu/arquivo/financial_regression.csv'

Salve o arquivo.

Execute o script no terminal ou no ambiente escolhido:

  python gerar_graficos.py

4. Análise dos Resultados

    Gráficos Gerados: O código salva os gráficos no mesmo diretório do script ou na pasta especificada. Os gráficos gerados são:
        gold_close_price_evolution.png: Evolução do preço de fechamento do ouro.
        gold_correlation_map.png: Mapa de correlação.
        gold_real_vs_predicted.png: Valores reais vs. previstos.
        gold_predictor_importance.png: Importância das variáveis preditivas.

    Baixar e visualizar: Abra os arquivos PNG com qualquer visualizador de imagens.

5. Interpretação

    Use os gráficos para entender padrões e insights sobre os dados.
    Avalie o desempenho do modelo de regressão com as métricas fornecidas no terminal: MSE e R².
