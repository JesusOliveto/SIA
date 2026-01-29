# Registro de Cambios (Changelog)

Este documento detalla secuencialmente las intervenciones realizadas en el proyecto para cumplir con los requerimientos, corregir errores y mejorar la calidad del software.

## 1. Auditoría Inicial y Testing
- **Diagnóstico:** Se detectó que la suite de tests (`pytest`) no se ejecutaba completamente por problemas de entorno y dependencias.
- **Acción:** Se configuró el entorno virtual (`.venv`) correctamente.
- **Resultado:** Se verificó la ejecución exitosa de **8 tests unitarios**, cubriendo carga de datos, lógica de los algoritmos K-Means (Loop y Numpy) y evaluación.

## 2. Mejoras de Visualización
- **Objetivo:** Mejorar la apreciación de los clusters más allá de métricas numéricas.
- **Acción:**
    - Se agregó la librería `plotly` a `requirements.txt`.
    - Se creó el módulo `src/visualization.py`.
    - Se implementaron dos nuevos gráficos interactivos:
        1.  **Mapa PCA 2D:** Proyección de los datos a 2 dimensiones para visualizar la separación espacial.
        2.  **Radar Chart de Centroides:** Gráfico radial para comparar las características promedio de cada cluster (perfil del vino).
    - Se integró una nueva sección "Visualización Profunda" en `app.py`.

## 3. Corrección de Despliegue (Deploy)
- **Error:** Fallo en el despliegue en la nube por incompatibilidad entre versiones antiguas de Streamlit (1.19) y nuevas de Altair (6.0).
- **Acción:** Se actualizaron y fijaron versiones modernas en `requirements.txt` (`streamlit>=1.50.0`, `altair`, etc.) para replicar el entorno local estable.

## 4. Explicación de Predicciones
- **Objetivo:** Hacer pedagógico el proceso de clasificación de un nuevo registro.
- **Acción:** Se modificó la interfaz de predicción en `app.py` para mostrar el proceso paso a paso:
    1.  **Normalización:** Comparativa de valores crudos vs normalizados (Z-Score).
    2.  **Cálculo de Distancias:** Tabla detallada de la distancia a cada centroide.
    3.  **Decisión:** Explicación textual de la asignación basada en la distancia mínima.

## 5. Limpieza de Datos
- **Objetivo:** Mejorar la calidad de los grupos eliminando ruido estadístico.
- **Acción:**
    - Se movió el dataset original a la carpeta `datasets/`.
    - Se creó un script de limpieza basado en el **Rango Intercuartil (IQR)**.
    - Se generó un nuevo dataset `datasets/whinequalityclean.arff` eliminando 630 registros atípicos (~18% de los datos).
    - Se reconfiguró `app.py` para utilizar este dataset depurado por defecto.

---
**Estado Final:** El proyecto cumple con todos los requerimientos académicos y técnicos, incluyendo mejoras significativas en usabilidad y robustez.
