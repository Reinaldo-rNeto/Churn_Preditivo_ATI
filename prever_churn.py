#!/usr/bin/env python3
"""Prevê a probabilidade de churn para uma lista de clientes.

Uso:
    python prever_churn.py --input exemplo_novos_clientes.csv --output previsoes.csv

O CSV de entrada precisa ter as mesmas colunas do dataset original
(exceto customerID, que é opcional, e Churn, que não deve estar presente).
"""
import argparse

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Prevê churn de clientes a partir de um CSV.")
    parser.add_argument("--input", required=True, help="CSV com os dados dos clientes")
    parser.add_argument("--output", default="previsoes.csv", help="Caminho do CSV de saída")
    parser.add_argument("--modelo", default="modelo_churn.joblib", help="Caminho do modelo treinado")
    args = parser.parse_args()

    pipeline = joblib.load(args.modelo)
    df = pd.read_csv(args.input)

    customer_ids = df["customerID"] if "customerID" in df.columns else pd.Series(range(len(df)), name="customerID")
    X = df.drop(columns=[c for c in ["customerID", "Churn"] if c in df.columns])

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)

    probabilidades = pipeline.predict_proba(X)[:, 1]
    previsoes = (probabilidades >= 0.5).astype(int)

    resultado = pd.DataFrame({
        "customerID": customer_ids,
        "probabilidade_churn": probabilidades.round(4),
        "previsao": ["Cancelou" if p == 1 else "Ficou" for p in previsoes],
    }).sort_values("probabilidade_churn", ascending=False)

    resultado.to_csv(args.output, index=False)
    print(f"{len(resultado)} clientes avaliados. Resultado salvo em {args.output}")
    print(f"Clientes com risco alto de churn (>50%): {(previsoes == 1).sum()}")


if __name__ == "__main__":
    main()
