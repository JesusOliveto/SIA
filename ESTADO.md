# ESTADO DEL PROYECTO

**Fecha de Última Auditoría:** 2026-02-12
**Estado:** FINALIZADO Y VERIFICADO
**Resultado:** LISTO PARA ENTREGA / MESA DE EXAMEN

---

## 1. Resumen de Auditoría
El proyecto ha sido auditado integralmente. Se han cumplido todos los requerimientos académicos del `readme.md` y se ha elevado el estándar de código con prácticas profesionales (type hinting, docstrings, logging, manejo de errores y UI pulida).

## 2. Mapa de Archivos del Proyecto
A continuación se detalla el propósito de cada archivo en el repositorio:

### Raíz
- **`app.py`**: Aplicación principal (Streamlit). Orquesta la carga de datos, selección de algoritmos, visualización y lógica de interacción con el usuario.
- **`clean_data.py`**: Script de utilidad para pre-procesar datos (limpieza de outliers, conversión ARFF). Se ejecuta de forma independiente.
- **`requirements.txt`**: Lista de dependencias de Python necesarias para ejecutar el proyecto (`numpy`, `pandas`, `streamlit`, `plotly`, `scipy`, `scikit-learn`).
- **`readme.md`**: Documento oficial con los requerimientos y consignas del TP Final de la cátedra.
- **`ESTADO.md`**: (Este archivo) Registro del estado actual del proyecto y auditoría de archivos.
- **`CAMBIOS.md`**: Historial cronológico de modificaciones y mejoras realizadas en el código.

### Módulo `src/` (Lógica del Negocio)
Contiene la implementación modular del sistema:

- **`kmeans_numpy.py`**: Implementación **vectorizada** de K-Means usando NumPy. Es la versión eficiente y recomendada para uso general.
- **`kmeans_loop.py`**: Implementación **no vectorizada** (bucles explícitos) de K-Means. Cumple con el requisito académico de demostrar la lógica "a pie". Útil para fines educativos.
- **`kmeans_sklearn.py`**: Wrapper (envoltura) de la implementación de `scikit-learn`. Sirve como línea base (benchmark) para comparar rendimiento y calidad.
- **`kmeans_base.py`**: Define la interfaz común (`Protocol`) que deben seguir todas las implementaciones de K-Means.
- **`data.py`**: Funciones para la carga, parsing (.arff) y normalización (Z-Score) de los datasets.
- **`visualization.py`**: Generación de gráficos interactivos con Plotly (PCA 2D y Radar Charts de centroides).
- **`evaluation.py`**: Lógica para correr benchmarks, ejecutar múltiples corridas y calcular métricas (Inercia, Silhouette, Tiempo).
- **`utils.py`**: Utilidades menores, principalmente manejo robusto de semillas aleatorias (`random_state`).

### Directorio `datasets/`
- **`winequality.arff`**: Dataset original crudo.
- **`whinequalityclean.arff`**: Dataset procesado y limpio (sin outliers).

### Directorio `tests/`
- **`verify_manual.py`**: Script de verificación rápida para asegurar que las implementaciones propias converjan correctamente en datos sintéticos.

## 3. Estado de Cumplimiento de Requerimientos

| Requerimiento | Estado | Evidencia / Archivo |
|---|---|---|
| **Lenguaje (Python)** | ✅ CUMPLIDO | Todo el proyecto en Python 3.10+. |
| **Implementación Propia (No Vectorizada)** | ✅ CUMPLIDO | `src/kmeans_loop.py` (con docstrings explicativos). |
| **Implementación Propia (Vectorizada)** | ✅ CUMPLIDO | `src/kmeans_numpy.py` (optimizado con broadcasting). |
| **Flexibilidad (k, dimensiones)** | ✅ CUMPLIDO | Implementaciones agnósticas a la dimensión de entrada. UI permite elegir `k`. |
| **Normalización de Datos** | ✅ CUMPLIDO | `src/data.py` aplica Z-Score automáticamente. |
| **Comparativa con Librería (Framework)** | ✅ CUMPLIDO | `src/kmeans_sklearn.py` integrado en la UI de comparativa. |
| **Interfaz Amigable (Documentación)** | ✅ CUMPLIDO | Streamlit App con pestañas por funcionalidad, explicaciones y gráficos interactivos. |
| **Predicción de Nuevos Registros** | ✅ CUMPLIDO | Pestaña "3. Predicción" en `app.py`. Muestra distancias y decisión final paso a paso. |

## 4. Notas Finales
El código se encuentra documentado en español (docstrings) para facilitar su defensa en la mesa de examen. Se ha verificado que no existe "cacheo" (session state) de los modelos entrenados en la predicción individual, forzando el re-entrenamiento para demostración en vivo, tal como solicitó el usuario.
