import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


file_path = '/mnt/data/financial_regression.csv'
df = pd.read_csv(file_path)


df['date'] = pd.to_datetime(df['date'], errors='coerce')


df_clean = df.dropna(subset=['gold close', 'gold open', 'gold high', 'gold low', 'gold volume'])


df_clean = df_clean.sort_values('date')


plt.figure(figsize=(12, 6))
plt.plot(df_clean['date'], df_clean['gold close'], label='Preço de Fechamento (Gold)', color='gold')
plt.title('Evolução do Preço de Fechamento do Ouro')
plt.xlabel('Data')
plt.ylabel('Preço (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/data/gold_close_price_evolution.png")
plt.show()


plt.figure(figsize=(10, 6))
correlation_matrix = df_clean[['gold open', 'gold high', 'gold low', 'gold close', 'gold volume']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Correlação das Variáveis Relacionadas ao Ouro")
plt.tight_layout()
plt.savefig("/mnt/data/gold_correlation_map.png")
plt.show()


variaveis_preditoras = ['gold open', 'gold high', 'gold low', 'gold volume']
variavel_alvo = 'gold close'
X = df_clean[variaveis_preditoras]
y = df_clean[variavel_alvo]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


modelo = LinearRegression()
modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Erro Quadrático Médio (MSE):", mse)
print("Coeficiente de Determinação (R²):", r2)


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='green', label='Previsões')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Linha Ideal')
plt.title('Valores Reais vs. Previstos (Preço de Fechamento do Ouro)')
plt.xlabel('Valor Real (Gold Close)')
plt.ylabel('Valor Previsto (Gold Close)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/data/gold_real_vs_predicted.png")
plt.show()


importancia = modelo.coef_
plt.figure(figsize=(10, 6))
plt.bar(variaveis_preditoras, importancia, color='orange')
plt.title('Importância das Variáveis Preditivas')
plt.ylabel('Coeficientes')
plt.tight_layout()
plt.savefig("/mnt/data/gold_predictor_importance.png")
plt.show()
