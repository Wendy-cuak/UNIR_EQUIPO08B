# Monitoreo de consumo de combustible y detección de anomalías en motores de riego

Proyecto de análisis de datos aplicado a motores de riego de la industria azucarera en Guatemala, utilizando datos provenientes de sensores IoT para estimar el consumo esperado de combustible, detectar anomalías operativas y clasificar alertas que apoyen la toma de decisiones.

## Descripción general

Este proyecto desarrolla un prototipo analítico orientado a transformar registros históricos de sensores IoT en información útil para el monitoreo operativo del consumo de combustible. El flujo integra preparación de datos, modelado predictivo, detección de anomalías por residuos y clasificación operativa de alertas, con visualización final en un dashboard.

El enfoque busca:

- Identificar desviaciones relevantes en el consumo de combustible.
- Clasificar eventos operativos en categorías interpretables.
- Facilitar la supervisión técnica y operativa mediante visualizaciones.

## Problema abordado

En la operación de motores de riego, el combustible representa un costo crítico y también un punto vulnerable frente a ineficiencias, fallos de medición o posibles irregularidades. Aunque los sensores IoT generan grandes volúmenes de información, su valor depende de convertir esos registros en alertas accionables.

Este repositorio implementa una solución analítica para:

- estimar el consumo esperado de cada equipo,
- comparar ese valor con el consumo observado,
- detectar desvíos atípicos,
- y priorizar alertas para revisión.

## Objetivo del proyecto

Construir un prototipo funcional de monitoreo de combustible que permita analizar datos históricos de motores de riego y sentar las bases para una futura implementación operativa con datos en tiempo cercano al real.

## Metodología

La lógica del proyecto sigue una adaptación de CRISP-DM y se estructura en cuatro fases:

### Fase 1. Limpieza y reglas de negocio

Se realiza la carga del dataset, validación de columnas, conversión de tipos y generación de variables derivadas.

Principales acciones:

- validación de columnas obligatorias,
- limpieza de fechas, horas y variables numéricas,
- eliminación de registros incompletos,
- creación de `Equipo_Activo`,
- construcción de `Consumo_Operativo`,
- generación de variables temporales,
- cálculo de `RPM_Rango`,
- tratamiento básico de valores físicamente improbables.

**Entrada principal:** archivo Excel con hoja `Data`.

**Salida:**

- `salida/fase1_dataset_limpio.csv`
- `salida/fase1_resumen_calidad.csv`

### Fase 2. Regresión supervisada para estimar consumo esperado

Se entrena un modelo de regresión para estimar el consumo operativo esperado a partir de variables operativas y temporales. Se comparan dos enfoques:

- **Ridge**
- **Random Forest Regressor**

Características del modelado:

- separación temporal de datos: 80% entrenamiento y 20% prueba,
- imputación de variables numéricas y categóricas,
- codificación one-hot para `Ceco`,
- escalado numérico en Ridge,
- selección del mejor modelo con base en métricas de error.

**Métricas evaluadas:**

- RMSE
- MAE
- R²
- MAPE

**Salidas:**

- `salida/fase2_predicciones.csv`
- `salida/fase2_comparacion_modelos.csv`
- `salida/fase2_metricas_modelo.txt`

### Fase 3. Detección de anomalías por residuo

La detección de anomalías se basa en la diferencia entre el consumo observado y el consumo esperado.

Variables generadas:

- `Residuo`
- `Residuo_Abs`
- `Residuo_Pct`
- `Umbral_Anomalia`
- `Es_Anomalia`
- `Direccion_Desvio`

El umbral de anomalía no es fijo. Se calcula por equipo (`Ceco`) mediante una regla robusta:

```text
Umbral_Anomalia = mediana(Residuo_Abs) + 3 × MAD
```

Si un equipo no cuenta con umbral específico, se usa un umbral global de respaldo.

**Salidas:**

- `salida/fase3_anomalias.csv`
- `salida/fase3_umbrales_por_ceco.csv`

### Fase 4. Clasificación operativa de alertas

Las anomalías detectadas se convierten en alertas operativas interpretables mediante reglas de negocio, complementadas con un clustering auxiliar.

Tipos de alerta implementados:

- `Normal`
- `Critica_Bateria_y_Consumo`
- `Consumo_Ineficiente_en_Ralenti`
- `Operacion_Exigente_Sobreconsumo`
- `Subconsumo_Atipico`
- `Anomalia_No_Clasificada`

Además, se asigna una prioridad operativa:

- **Alta**
- **Media**
- **Baja**

Como apoyo exploratorio, se aplica **KMeans** únicamente sobre registros anómalos para perfilar subgrupos mediante:

- `Residuo_Abs`
- `Residuo_Pct`
- `Velocidad`
- `RPM_Promedio`
- `Bateria`

**Salidas:**

- `salida/fase4_alertas_clasificadas.csv`
- `salida/fase4_resumen_alertas.csv`

## Arquitectura del flujo

```text
Sensores IoT / archivos históricos
        ↓
Preparación y validación de datos
        ↓
Modelado predictivo del consumo esperado
        ↓
Detección de anomalías por residuo
        ↓
Clasificación operativa de alertas
        ↓
Visualización en Power BI
```

## Variables principales

Entre las variables de entrada más relevantes se encuentran:

- `Ceco`
- `Fecha`
- `Hora`
- `Consumo_Total`
- `Bateria`
- `Velocidad`
- `RPM_Promedio`
- `RPM_Min`
- `RPM_Max`

Variables derivadas destacadas:

- `Equipo_Activo`
- `Consumo_Operativo`
- `RPM_Rango`
- `DiaSemana`
- `Mes`
- `DiaMes`
- `EsFinSemana`
- `Consumo_Esperado`
- `Residuo`
- `Residuo_Abs`
- `Residuo_Pct`
- `Umbral_Anomalia`
- `Es_Anomalia`
- `Direccion_Desvio`
- `Tipo_Alerta_Regla`
- `Prioridad`
- `Cluster_Anomalia`

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Power BI
- SQL

## Estructura esperada del repositorio

```text
.
├── code_completo.ipynb
├── README.md
└── salida/
    ├── fase1_dataset_limpio.csv
    ├── fase1_resumen_calidad.csv
    ├── fase2_predicciones.csv
    ├── fase2_comparacion_modelos.csv
    ├── fase2_metricas_modelo.txt
    ├── fase3_anomalias.csv
    ├── fase3_umbrales_por_ceco.csv
    ├── fase4_alertas_clasificadas.csv
    └── fase4_resumen_alertas.csv
```

## Cómo ejecutar el notebook

1. Abrir `NUEVO_SCRIPT.ipynb` en Jupyter Notebook o Google Colab.
2. Ajustar la ruta del archivo de entrada en la celda de configuración de la Fase 1.
3. Verificar que el archivo Excel contenga la hoja `Data`.
4. Ejecutar las celdas en orden, desde la Fase 1 hasta la Fase 4.
5. Revisar la carpeta `salida/` para consultar los archivos generados.

## Requisitos mínimos

Instalar las librerías necesarias:

```bash
pip install pandas numpy scikit-learn openpyxl jupyter
```

## Casos de uso

Este prototipo puede apoyar tareas como:

- monitoreo de consumo por equipo,
- detección de sobreconsumo y subconsumo atípico,
- priorización de alertas operativas,
- análisis de patrones por motor o centro de costo,
- construcción de dashboards para supervisión técnica.

## Visualización

Los resultados pueden integrarse en un dashboard en Power BI para:

- revisar consumo horario,
- comparar consumo vs RPM,
- segmentar por equipo,
- visualizar alertas,
- seguir indicadores operativos.

## Limitaciones

- El sistema debe interpretarse como **apoyo a la decisión**, no como mecanismo automático de sanción.
- La calidad del resultado depende de la calidad del sensor y de la consistencia del registro histórico.
- Variables operativas adicionales, como caudal o volumen de agua bombeada, podrían mejorar la capacidad explicativa del modelo.
- El clustering auxiliar no reemplaza las reglas de negocio; solo complementa el análisis exploratorio.

## Líneas futuras

- integrar más variables operativas e hidráulicas,
- incorporar nuevas fuentes de datos,
- evaluar modelos adicionales,
- automatizar la ingestión de datos vía API,
- desplegar alertas en tiempo cercano al real,
- robustecer el dashboard para uso operativo continuo.

## Autoría académica

Trabajo desarrollado en el contexto del Máster Universitario en Análisis y Visualización de Datos Masivos.

**Equipo 08B**

- Silmert Rony Becerra Montoya
- Wendy Jimena Espinoza Bustos
- Liliana López Martínez
- José Ignacio Reyes Vicente


Este README resume el propósito, la arquitectura y el flujo analítico implementado en el notebook del proyecto. Puede adaptarse según la estructura final del repositorio, la disponibilidad de datos o la publicación del dashboard.
