# Registro de Cambios (Changelog)

Este documento detalla secuencialmente las intervenciones realizadas en el proyecto para cumplir con los requerimientos, corregir errores y mejorar la calidad del software.

## 1. Auditoría Inicial y Testing
- **Acción:** Se configuró el entorno virtual (`.venv`) y se verificaron 8 tests unitarios.

## 2. Mejoras de Visualización
- **Acción:** Se implementaron **Mapa PCA 2D** y **Radar Chart** interactivos.

## 3. Corrección de Despliegue (Deploy)
- **Acción:** Se fijaron versiones estables en `requirements.txt`.

## 4. Explicación de Predicciones
- **Acción:** Se modificó la interfaz para mostrar **Normalización**, **Distancias** y **Decisión** paso a paso.

## 5. Limpieza de Datos
- **Acción:** Nuevo dataset `whinequalityclean.arff` (630 outliers removidos).

## 6. Flexibilidad de Interfaz
- **Acción:**
    - **Selector de Dataset:** Original vs Limpio.
    - **Modos de Ejecución:** Rendimiento (Promedio) y Depuración (Logs paso a paso).

## 7. Método del Codo y Documentación UI
- **Objetivo:** Facilitar la elección de K y mejorar la usabilidad.
- **Acción:**
    - Se insertó la sección **"Determinación de K óptimo"** que grafica Inercia vs K.
    - Se añadieron descripciones explicativas ("captions") debajo de cada título de sección para guiar al usuario.

---
**Estado Final:** El proyecto cumple con todos los requerimientos académicos y técnicos.
