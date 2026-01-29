# Registro de Cambios (Changelog)

Este documento detalla secuencialmente las intervenciones realizadas en el proyecto para cumplir con los requerimientos, corregir errores y mejorar la calidad del software.

## 1. Auditoría Inicial y Testing
- **Diagnóstico:** Se detectó que la suite de tests (`pytest`) no se ejecutaba completamente por problemas de entorno y dependencias.
- **Acción:** Se configuró el entorno virtual (`.venv`) correctamente y se verificaron 8 tests unitarios.

## 2. Mejoras de Visualización
- **Objetivo:** Mejorar la apreciación de los clusters.
- **Acción:**
    - Se agregó `plotly`.
    - Se implementaron **Mapa PCA 2D** y **Radar Chart**.

## 3. Corrección de Despliegue (Deploy)
- **Error:** Fallo en el despliegue por versiones obsoletas.
- **Acción:** Se fijaron versiones modernass en `requirements.txt`.

## 4. Explicación de Predicciones
- **Objetivo:** Hacer pedagógico el proceso de clasificación.
- **Acción:** Se modificó la interfaz para mostrar **Normalización**, **Distancias** y **Decisión** paso a paso.

## 5. Limpieza de Datos
- **Objetivo:** Mejorar la calidad estadistica.
- **Acción:**
    - Script de limpieza IQR (630 outliers removidos).
    - Nuevo dataset `whinequalityclean.arff`.

## 6. Flexibilidad de Interfaz
- **Objetivo:** Permitir experimentación y análisis profundo.
- **Acción:**
    - **Selector de Dataset:** Permite elegir entre datos originales o limpios.
    - **Modos de Ejecución:**
        - *Rendimiento:* Promedia 5 corridas para evaluar velocidad y estabilidad.
        - *Depuración:* Ejecuta una vez mostrando logs internos del algoritmo iteración a iteración.

---
**Estado Final:** El proyecto cumple con todos los requerimientos académicos y técnicos, incluyendo mejoras significativas en usabilidad y robustez.
