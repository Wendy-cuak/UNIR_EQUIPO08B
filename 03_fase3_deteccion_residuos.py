#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 3: Detección de anomalías por residuo
------------------------------------------
Objetivo:
    - Comparar consumo real vs consumo esperado.
    - Calcular residuo absoluto y relativo.
    - Definir umbrales robustos basados en MAD (Median Absolute Deviation).
    - Etiquetar registros anómalos.

Entradas:
    - salida/fase2_predicciones.csv

Salidas:
    - salida/fase3_anomalias.csv
    - salida/fase3_umbrales_por_ceco.csv

Criterio propuesto:
    residuo = consumo_real - consumo_esperado
    anomalia si |residuo| > mediana(|residuo|) + 3 * MAD(|residuo|)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def mad(series: pd.Series) -> float:
    med = np.median(series)
    return float(np.median(np.abs(series - med)))


def calcular_umbrales(df: pd.DataFrame) -> pd.DataFrame:
    filas = []

    for ceco, g in df.groupby("Ceco"):
        abs_res = g["Residuo_Abs"].dropna()
        if len(abs_res) == 0:
            continue
        med = float(np.median(abs_res))
        mad_val = mad(abs_res)
        umbral = med + 3 * mad_val

        filas.append({
            "Ceco": ceco,
            "Mediana_Residuo_Abs": med,
            "MAD_Residuo_Abs": mad_val,
            "Umbral_Anomalia": umbral
        })

    return pd.DataFrame(filas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 3: detección de anomalías por residuo")
    parser.add_argument("--input", default="salida/fase2_predicciones.csv", help="Predicciones de fase 2")
    parser.add_argument("--outdir", default="salida", help="Carpeta de salida")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    df["Residuo"] = df["Consumo_Operativo"] - df["Consumo_Esperado"]
    df["Residuo_Abs"] = np.abs(df["Residuo"])
    df["Residuo_Pct"] = (
        df["Residuo_Abs"] / np.clip(np.abs(df["Consumo_Esperado"]), 1e-6, None)
    ) * 100

    umbrales = calcular_umbrales(df)
    df = df.merge(umbrales[["Ceco", "Umbral_Anomalia"]], on="Ceco", how="left")

    # Fallback global en caso de umbral faltante
    umbral_global = float(np.median(df["Residuo_Abs"].dropna()) + 3 * mad(df["Residuo_Abs"].dropna()))
    df["Umbral_Anomalia"] = df["Umbral_Anomalia"].fillna(umbral_global)

    df["Es_Anomalia"] = (df["Residuo_Abs"] > df["Umbral_Anomalia"]).astype(int)
    df["Direccion_Desvio"] = np.where(df["Residuo"] > 0, "Sobreconsumo", "Subconsumo")

    salida_anom = outdir / "fase3_anomalias.csv"
    salida_umb = outdir / "fase3_umbrales_por_ceco.csv"

    df.to_csv(salida_anom, index=False, encoding="utf-8-sig")
    umbrales.to_csv(salida_umb, index=False, encoding="utf-8-sig")

    tasa = df["Es_Anomalia"].mean() * 100
    print("Detección de anomalías completada.")
    print(f"Anomalías detectadas: {df['Es_Anomalia'].sum():,}")
    print(f"Tasa de anomalía: {tasa:.2f}%")
    print(f"Archivo de anomalías: {salida_anom}")
    print(f"Archivo de umbrales: {salida_umb}")


if __name__ == "__main__":
    main()
