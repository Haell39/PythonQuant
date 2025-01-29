


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregamento e Análise Exploratória dos Dados
df = pd.read_csv('Dataset.csv')

# Verificando valores ausentes
print("Valores ausentes por coluna:")
print(df.isnull().sum())

# Tratando valores ausentes
# Removendo linhas com tempo no site negativo
df = df[df['Tempo no Site (min)'] >= 0]

# Preenchendo valores ausentes
df['Idade'].fillna(df['Idade'].mean(), inplace=True)
df['Renda Anual (em $)'].fillna(df['Renda Anual (em $)'].mean(), inplace=True)
df['Gênero'].fillna(df['Gênero'].mode()[0], inplace=True)
df['Anúncio Clicado'].fillna(df['Anúncio Clicado'].mode()[0], inplace=True)

# Análise estatística básica
print("\nEstatísticas descritivas:")
print(df.describe())

# Visualizações
plt.figure(figsize=(15, 10))

# Distribuição de idade por compra
plt.subplot(2, 2, 1)
sns.boxplot(x='Compra (0 ou 1)', y='Idade', data=df)
plt.title('Distribuição de Idade por Compra')

# Distribuição de renda por compra
plt.subplot(2, 2, 2)
sns.boxplot(x='Compra (0 ou 1)', y='Renda Anual (em $)', data=df)
plt.title('Distribuição de Renda por Compra')

# Tempo no site por compra
plt.subplot(2, 2, 3)
sns.boxplot(x='Compra (0 ou 1)', y='Tempo no Site (min)', data=df)
plt.title('Distribuição de Tempo no Site por Compra')

# Relação entre gênero e compra
plt.subplot(2, 2, 4)
sns.countplot(data=df, x='Gênero', hue='Compra (0 ou 1)')
plt.title('Distribuição de Compras por Gênero')

plt.tight_layout()
plt.show()

# 2. Pré-processamento dos Dados
# Codificação de variáveis categóricas
le = LabelEncoder()
df['Gênero_encoded'] = le.fit_transform(df['Gênero'])
df['Anúncio_Clicado_encoded'] = le.fit_transform(df['Anúncio Clicado'])

# Seleção de features
X = df[['Idade', 'Renda Anual (em $)', 'Gênero_encoded', 
        'Tempo no Site (min)', 'Anúncio_Clicado_encoded']]
y = df['Compra (0 ou 1)']

# Divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Construção e Avaliação do Modelo
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Previsões
y_pred = rf_model.predict(X_test_scaled)

# Avaliação do modelo
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# 4. Interpretação do Modelo
# Importância das features
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Importância das Features no Modelo')
plt.show()

# Matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

```

Baseado na análise realizada, aqui estão os principais pontos a serem avaliados no trabalho do estagiário:

1. Análise Exploratória:
- O código trata adequadamente os valores ausentes e inconsistentes no dataset
- Foram geradas visualizações importantes mostrando a relação entre as variáveis e a compra
- As estatísticas descritivas permitem entender a distribuição dos dados

2. Pré-processamento:
- As variáveis categóricas foram codificadas adequadamente
- Os dados foram normalizados usando StandardScaler
- A divisão treino-teste foi realizada corretamente

3. Modelagem:
- Foi utilizado o Random Forest, que é um bom modelo para este tipo de problema
- O modelo foi avaliado usando métricas apropriadas (classification_report)
- Foi gerada a matriz de confusão para visualizar os erros do modelo

4. Interpretação:
- A importância das features foi analisada e visualizada
- O código gera visualizações que ajudam a entender o comportamento do modelo

Pontos extras implementados:
- Visualizações gráficas usando seaborn
- Análise da importância das features
- Matriz de confusão
- Tratamento adequado de valores ausentes

