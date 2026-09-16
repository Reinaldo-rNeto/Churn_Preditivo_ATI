"""Treina o modelo de produção (Regressão Logística) com todo o histórico disponível
e salva o pipeline completo (pré-processamento + modelo) em modelo_churn.joblib.

As métricas reportadas no README/notebook vêm da avaliação com held-out test set
(ver notebook_modelo_preditivo.ipynb) -- este script re-treina no dataset completo
antes de "shippar", prática padrão pra colocar o melhor modelo possível em produção.
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

customer_ids = df["customerID"]
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat_cols = [c for c in X.columns if c not in num_cols]

preprocessador = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

pipeline = Pipeline([
    ("preprocessador", preprocessador),
    ("modelo", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])

pipeline.fit(X, y)

joblib.dump(pipeline, "modelo_churn.joblib")
print("Modelo salvo em modelo_churn.joblib")

# Gera um CSV de exemplo com clientes reais (sem a coluna Churn) pra demonstrar o prever_churn.py
exemplo = df.drop(columns=["Churn"]).sample(8, random_state=RANDOM_STATE)
exemplo.to_csv("exemplo_novos_clientes.csv", index=False)
print("Exemplo de entrada salvo em exemplo_novos_clientes.csv")
