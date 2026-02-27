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


## 9. Limpieza de Datos (Data Cleaning)
- **Objetivo:** Mejorar la calidad del dataset eliminando outliers para obtener resultados de clustering más significativos.
- **Acción:**
    - Se creó el script `clean_data.py`.
    - Se implementó un filtro basado en el **Rango Intercuartílico (IQR)** para detectar y eliminar valores atípicos en las variables numéricas.
    - Se generó un nuevo archivo `whinequalityclean.arff` con los datos depurados.
    - Se actualizó `app.py` para cargar el nuevo dataset por defecto, mejorando la calidad de los gráficos y métricas.

## 10. Mejoras en la Interfaz de Usuario (UI)
- **Objetivo:** Mejorar la experiencia del usuario y la presentación visual de los resultados.
- **Acción:**
    - Se eliminaron los gráficos de barras de "Iteraciones" y "Tiempo" debido a su irrelevancia para el análisis de clustering.
    - Se implementó un nuevo gráfico interactivo de **Distribución de Calidad** para el dataset limpio, permitiendo visualizar la distribución de las etiquetas de calidad.
    - Se mejoró la presentación de los resultados de evaluación, ocultando las columnas "Iteraciones" y "Tiempo" de la tabla resumen para mayor claridad.


## 11. Funcionalidades Adicionales
- **Objetivo:** Implementar funcionalidades faltantes detectadas.
- **Acción:**
    - **Explorador de Datos:** Se agregó una nueva pestaña para inspeccionar datos crudos vs normalizados y ver estadísticas descriptivas por atributo.
    - **Selección de Atributos:** Se implementó un multiselect en el sidebar que permite filtrar qué features utilizan los algoritmos.
    - **Visualización Directa:** Se añadió lógica para graficar Scatter 2D y 3D directos (sin PCA) cuando se seleccionan 2 o 3 atributos respectivamente.

## 12. Documentación
- **Objetivo:** Mejorar la documentación del proyecto.
- **Acción:**
    - Se actualizó la documentación de todos los archivos del proyecto.
    - Se corrigieron los docstrings y se añadieron comentarios exhaustivos.

## 13. Calidad Promedio por Cluster
- **Objetivo:** Mostrar la calidad promedio de los vinos en cada cluster.
- **Acción:**
    - Se agregó una nueva pestaña para mostrar la calidad promedio de los vinos en cada cluster.

## 14. Min-Max
- **Objetivo:** Implementar la normalización Min-Max en el proyecto.
- **Acción:**
    - Se agregó la normalización Min-Max en el proyecto.

## 15. Z-Score -> escalado optimo
- **Objetivo:** Implementar la normalización Z-Score en el proyecto.
- **Acción:**
    - Se agregó la normalización Z-Score en el proyecto.