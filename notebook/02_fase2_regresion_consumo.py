#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 2: Regresión supervisada para estimar consumo esperado
-----------------------------------------------------------
Objetivo:
    - Entrenar un modelo supervisado que estime el consumo esperado.
    - Utilizar variables operativas y temporales.
    - Guardar predicciones y métricas para la fase de residuos.

Entradas:
    - salida/fase1_dataset_limpio.csv

Salidas:
    - salida/fase2_predicciones.csv
    - salida/fase2_metricas_modelo.txt

Notas metodológicas:
    - Se usa una partición temporal: train = primeras fechas, test = últimas fechas.
    - Se entrena sobre registros con Equipo_Activo = 1 y Consumo_Operativo no nulo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df[(df["Equipo_Activo"] == 1) & (df["Consumo_Operativo"].notna())].copy()
    df = df.sort_values(["Fecha", "Hora"]).reset_index(drop=True)
    return df


def split_temporal(df: pd.DataFrame, proporcion_train: float = 0.8):
    corte = int(len(df) * proporcion_train)
    train = df.iloc[:corte].copy()
    test = df.iloc[corte:].copy()
    return train, test


def construir_modelo(num_features, cat_features):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocess", pre),
        ("model", model),
    ])
    return pipe


def metricas_regresion(y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE_pct": mape
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2: regresión supervisada")
    parser.add_argument("--input", default="salida/fase1_dataset_limpio.csv", help="CSV limpio de fase 1")
    parser.add_argument("--outdir", default="salida", help="Carpeta de salida")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = preparar_datos(df)

    features_num = [
        "Hora", "Bateria", "Velocidad", "RPM_Promedio",
        "RPM_Min", "RPM_Max", "RPM_Rango",
        "DiaSemana", "Mes", "DiaMes", "EsFinSemana"
    ]
    features_cat = ["Ceco"]
    target = "Consumo_Operativo"

    train, test = split_temporal(df, proporcion_train=0.8)

    X_train = train[features_num + features_cat]
    y_train = train[target]
    X_test = test[features_num + features_cat]
    y_test = test[target]

    pipe = construir_modelo(features_num, features_cat)
    pipe.fit(X_train, y_train)

    test["Consumo_Esperado"] = pipe.predict(X_test)
    met = metricas_regresion(y_test, test["Consumo_Esperado"])

    salida_pred = outdir / "fase2_predicciones.csv"
    test.to_csv(salida_pred, index=False, encoding="utf-8-sig")

    salida_metricas = outdir / "fase2_metricas_modelo.txt"
    with open(salida_metricas, "w", encoding="utf-8") as f:
        f.write("Métricas del modelo de regresión\n")
        f.write("--------------------------------\n")
        for k, v in met.items():
            f.write(f"{k}: {v:.6f}\n")

    print("Modelo entrenado y evaluado.")
    print(f"Predicciones guardadas en: {salida_pred}")
    print(f"Métricas guardadas en: {salida_metricas}")
    for k, v in met.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
