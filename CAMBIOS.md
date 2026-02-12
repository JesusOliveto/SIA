# Registro de Cambios (Changelog)

Este documento detalla secuencialmente las intervenciones realizadas en el proyecto para cumplir con los requerimientos, corregir errores y mejorar la calidad del software.

## 1. Auditoría Inicial y Testing
- **Acción:** Se verificaron 8 tests unitarios.

## 2. Mejoras de Visualización
- **Acción:** `plotly` para PCA 2D y Radar Charts.

## 3. Corrección de Despliegue (Deploy)
- **Acción:** Versiones fijas en `requirements.txt`.

## 4. Explicación de Predicciones
- **Acción:** Desglose paso a paso (Normalización -> Distancias).

## 5. Limpieza de Datos
- **Acción:** Dataset limpio de outliers (IQR).

## 6. Flexibilidad de Interfaz
- **Acción:** Selector de Dataset y Modos (Rendimiento/Depuración).

## 7. Método del Codo y Mejora UI
- **Objetivo:** Facilitar la elección de K con alta precisión.
- **Acción:**
    - Se implementó gráfico interactivo (Plotly) de Inercia vs K.
    - Se amplió el rango de evaluación hasta **K=30**.
    - Se añadieron marcadores visuales para identificar mejor el codo.

## 8. Refactorización y Documentación
- **Objetivo:** Mejorar la legibilidad, mantenibilidad y robustez del código.
- **Acción:**
    - **Comentarios Exhaustivos:** Se documentaron todas las funciones, métodos y clases.
    - **Tipado Estricto:** Se corrigieron y completaron las anotaciones de tipos (`typing`).
    - **Docstrings:** Se estandarizaron los docstrings siguiendo el formato NumPy/Google.
    - **Corrección de Errores:** Se solucionaron errores de lógica en `KMeansLoop` (asignación de etiquetas) y `KMeansNumpy` (manejo de clusters vacíos).
    - **Consistencia:** Se unificó el uso de `random_state` y se mejoró el manejo de `verbose`.


---
**Estado Final:** El proyecto cumple con todos los requerimientos académicos y técnicos.
