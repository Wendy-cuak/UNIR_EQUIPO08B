#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 4: Clasificación operativa de alertas
------------------------------------------
Objetivo:
    - Convertir anomalías estadísticas en alertas operativas interpretables.
    - Combinar reglas de negocio con clustering auxiliar.
    - Priorizar alertas para seguimiento.

Entradas:
    - salida/fase3_anomalias.csv

Salidas:
    - salida/fase4_alertas_clasificadas.csv
    - salida/fase4_resumen_alertas.csv

Lógica:
    1) Si no es anomalía -> estado normal.
    2) Si es anomalía:
        - batería baja + residuo alto => alerta crítica
        - velocidad 0 + rpm > 0 + sobreconsumo => alerta por ralentí/consumo ineficiente
        - velocidad alta + residuo alto => alerta por operación exigente
    3) Clustering auxiliar (KMeans) para perfilar subtipos de anomalías.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def clasificar_reglas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Tipo_Alerta_Regla"] = "Normal"

    mask_anom = df["Es_Anomalia"] == 1

    crit = mask_anom & (df["Bateria"] < 20) & (df["Residuo_Abs"] > df["Umbral_Anomalia"] * 1.2)
    ralenti = mask_anom & (df["Velocidad"] == 0) & (df["RPM_Promedio"] > 0) & (df["Direccion_Desvio"] == "Sobreconsumo")
    exigente = mask_anom & (df["Velocidad"] > df["Velocidad"].median()) & (df["Direccion_Desvio"] == "Sobreconsumo")
    subcons = mask_anom & (df["Direccion_Desvio"] == "Subconsumo")

    df.loc[crit, "Tipo_Alerta_Regla"] = "Critica_Bateria_y_Consumo"
    df.loc[ralenti, "Tipo_Alerta_Regla"] = "Consumo_Ineficiente_en_Ralenti"
    df.loc[exigente, "Tipo_Alerta_Regla"] = "Operacion_Exigente_Sobreconsumo"
    df.loc[subcons, "Tipo_Alerta_Regla"] = "Subconsumo_Atipico"
    df.loc[mask_anom & (df["Tipo_Alerta_Regla"] == "Normal"), "Tipo_Alerta_Regla"] = "Anomalia_No_Clasificada"

    prioridad = {
        "Critica_Bateria_y_Consumo": "Alta",
        "Consumo_Ineficiente_en_Ralenti": "Alta",
        "Operacion_Exigente_Sobreconsumo": "Media",
        "Subconsumo_Atipico": "Media",
        "Anomalia_No_Clasificada": "Media",
        "Normal": "Baja",
    }
    df["Prioridad"] = df["Tipo_Alerta_Regla"].map(prioridad)
    return df


def clustering_auxiliar(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    df = df.copy()
    anom = df[df["Es_Anomalia"] == 1].copy()

    if len(anom) < n_clusters:
        df["Cluster_Anomalia"] = -1
        return df

    features = ["Residuo_Abs", "Residuo_Pct", "Velocidad", "RPM_Promedio", "Bateria"]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    X = pipe.fit_transform(anom[features])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    anom["Cluster_Anomalia"] = km.fit_predict(X)

    df["Cluster_Anomalia"] = -1
    df.loc[anom.index, "Cluster_Anomalia"] = anom["Cluster_Anomalia"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 4: clasificación operativa de alertas")
    parser.add_argument("--input", default="salida/fase3_anomalias.csv", help="Archivo de anomalías")
    parser.add_argument("--outdir", default="salida", help="Carpeta de salida")
    parser.add_argument("--clusters", default=3, type=int, help="Número de clusters auxiliares")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = clasificar_reglas(df)
    df = clustering_auxiliar(df, n_clusters=args.clusters)

    resumen = (
        df.groupby(["Tipo_Alerta_Regla", "Prioridad"], dropna=False)
          .size()
          .reset_index(name="Cantidad")
          .sort_values("Cantidad", ascending=False)
    )

    salida_alertas = outdir / "fase4_alertas_clasificadas.csv"
    salida_resumen = outdir / "fase4_resumen_alertas.csv"

    df.to_csv(salida_alertas, index=False, encoding="utf-8-sig")
    resumen.to_csv(salida_resumen, index=False, encoding="utf-8-sig")

    print("Clasificación operativa completada.")
    print(f"Alertas guardadas en: {salida_alertas}")
    print(f"Resumen guardado en: {salida_resumen}")


if __name__ == "__main__":
    main()
