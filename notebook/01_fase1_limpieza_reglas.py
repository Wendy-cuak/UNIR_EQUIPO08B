#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 1: Limpieza y reglas de negocio
------------------------------------
Objetivo:
    - Leer el archivo Excel original.
    - Estandarizar tipos de datos.
    - Crear variables derivadas coherentes con el caso de uso.
    - Aplicar reglas de negocio para distinguir actividad operativa de inactividad.
    - Generar un dataset limpio para las fases siguientes.

Entradas:
    - Dataset-Sensores.xlsx (hoja: Data)

Salidas:
    - salida/fase1_dataset_limpio.csv
    - salida/fase1_resumen_calidad.csv

Reglas de negocio propuestas:
    1) Si RPM_Promedio = 0 y Velocidad = 0, el registro se considera inactivo.
    2) Si RPM_Promedio > 0 o Velocidad > 0, el registro se considera operativo.
    3) El modelo de consumo esperado debe entrenarse sobre consumo operativo,
       no sobre periodos totalmente inactivos.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def cargar_datos(ruta_excel: Path, hoja: str = "Data") -> pd.DataFrame:
    df = pd.read_excel(ruta_excel, sheet_name=hoja)
    return df


def limpiar_y_transformar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Estandarización de columnas esperadas
    columnas_esperadas = [
        "Ceco", "Fecha", "Hora", "Consumo_Total", "Bateria",
        "Velocidad", "RPM_Promedio", "RPM_Min", "RPM_Max"
    ]
    faltantes = [c for c in columnas_esperadas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas esperadas: {faltantes}")

    # Tipos de dato
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Hora"] = pd.to_numeric(df["Hora"], errors="coerce")
    for c in ["Consumo_Total", "Bateria", "Velocidad", "RPM_Promedio", "RPM_Min", "RPM_Max"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remover filas sin fecha/hora/ceco
    df = df.dropna(subset=["Fecha", "Hora", "Ceco"]).copy()

    # Acotar hora a 0-23 cuando corresponda
    df = df[(df["Hora"] >= 0) & (df["Hora"] <= 23)].copy()

    # Reglas de negocio de actividad
    df["Equipo_Activo"] = (
        (df["RPM_Promedio"].fillna(0) > 0) |
        (df["Velocidad"].fillna(0) > 0)
    ).astype(int)

    # Consumo operativo: solo en ventanas donde hay evidencia de operación
    df["Consumo_Operativo"] = np.where(
        df["Equipo_Activo"] == 1,
        df["Consumo_Total"],
        np.nan
    )

    # Variables temporales útiles para modelado
    df["DiaSemana"] = df["Fecha"].dt.dayofweek
    df["Mes"] = df["Fecha"].dt.month
    df["DiaMes"] = df["Fecha"].dt.day
    df["EsFinSemana"] = df["DiaSemana"].isin([5, 6]).astype(int)

    # Variables auxiliares de consistencia
    df["RPM_Rango"] = df["RPM_Max"] - df["RPM_Min"]
    df["Bateria_Baja"] = (df["Bateria"] < 20).astype(int)

    # Tratamiento simple de outliers físicos imposibles
    df.loc[df["Bateria"] < 0, "Bateria"] = np.nan
    df.loc[df["Bateria"] > 100, "Bateria"] = np.nan
    for c in ["Velocidad", "RPM_Promedio", "RPM_Min", "RPM_Max", "Consumo_Total"]:
        df.loc[df[c] < 0, c] = np.nan

    # Orden temporal
    df = df.sort_values(["Ceco", "Fecha", "Hora"]).reset_index(drop=True)
    return df


def generar_resumen_calidad(df: pd.DataFrame) -> pd.DataFrame:
    resumen = pd.DataFrame({
        "variable": df.columns,
        "nulos": df.isna().sum().values,
        "pct_nulos": (df.isna().mean().values * 100).round(2),
    })
    return resumen.sort_values("pct_nulos", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1: limpieza y reglas de negocio")
    parser.add_argument("--excel", default="Dataset-Sensores.xlsx", help="Ruta al Excel fuente")
    parser.add_argument("--hoja", default="Data", help="Nombre de la hoja")
    parser.add_argument("--outdir", default="salida", help="Carpeta de salida")
    args = parser.parse_args()

    ruta_excel = Path(args.excel)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = cargar_datos(ruta_excel, args.hoja)
    limpio = limpiar_y_transformar(df)
    resumen = generar_resumen_calidad(limpio)

    limpio.to_csv(outdir / "fase1_dataset_limpio.csv", index=False, encoding="utf-8-sig")
    resumen.to_csv(outdir / "fase1_resumen_calidad.csv", index=False, encoding="utf-8-sig")

    print("Proceso completado.")
    print(f"Filas procesadas: {len(limpio):,}")
    print(f"Equipos (CECO) únicos: {limpio['Ceco'].nunique():,}")
    print(f"Dataset limpio guardado en: {outdir / 'fase1_dataset_limpio.csv'}")
    print(f"Resumen de calidad guardado en: {outdir / 'fase1_resumen_calidad.csv'}")


if __name__ == "__main__":
    main()
