STATUS
Estructura creada. Implementados loader + normalización + K-Means (loop y numpy) + wrapper sklearn. Tests básicos pasaron.

---

## Contexto del TP
Materia: Sistemas Inteligentes Artificiales (SIA)
Tema: Implementación de K-Medias (K-Means)
Dataset provisto: `winequality.arff` (features numéricas + atributo `class`)

---

## Requisitos (según readme.md)
- Implementar K-Means **propio** en 2 variantes:
	- **No vectorizado** (loops)
	- **Vectorizado** (operaciones matriciales)
- No usar librerías/frameworks que ya contengan K-Means para las implementaciones propias.
- Sí se permite usar soporte matemático (p.ej. Numpy) y estadístico.
- Implementación **flexible**:
	- Permitir setear `k`
	- Soportar cualquier cantidad de ejemplos y atributos
- **Normalizar** los datos (evitar escalas distintas).
- Comparativa con una implementación de terceros (p.ej. scikit-learn), idealmente con:
	- Múltiples valores de `k`
	- Múltiples ejecuciones
	- Comparación gráfica
- Interfaz amigable y capacidad de “predecir” el cluster de un registro **nuevo**.
- La app debe correr sin errores + UI funcional (se valora alguna funcionalidad extra).

---

## Decisiones técnicas (propuesta)
- Lenguaje: **Python**.
- Implementaciones:
	1) `kmeans_loop`: propia no vectorizada.
	2) `kmeans_numpy`: propia vectorizada.
	3) `kmeans_sklearn`: baseline con librería para comparativa (solo para esta variante).
- UI: **una sola** aplicación **Streamlit** con selector de implementación (viable y recomendable para no duplicar lógica).
- Lectura del dataset ARFF: `scipy.io.arff` (alternativa: `liac-arff`).
- Normalización: por defecto **z-score** (media 0, desvío 1). Alternativa: min-max.
- Métricas y comparativa:
	- Tiempo de entrenamiento (wall time)
	- Inercia / SSE (sum of squared distances)
	- Silhouette score (opcional, útil para comparar k)
- Reproducibilidad: `random_state` controlable + repetir `n_runs`.

---

## Estructura propuesta del proyecto
Objetivo: separar núcleo (algoritmo) de la UI.

- `app.py` (Streamlit)
- `src/`
	- `data.py` (carga ARFF, split features/label, normalización)
	- `kmeans_base.py` (tipos + API común)
	- `kmeans_loop.py` (no vectorizado)
	- `kmeans_numpy.py` (vectorizado)
	- `kmeans_sklearn.py` (baseline)
	- `evaluation.py` (métricas, corridas múltiples, gráficos)
	- `utils.py` (semillas, validaciones)
- `requirements.txt`
- `README.md` (cómo correr local + deploy)
- `report/` (borradores/figuras para el PDF)

---

## API mínima esperada (para ambas implementaciones propias)
- `fit(X) -> self`
- `predict(X) -> labels`
- `centroids_` (centroides finales)
- `labels_` (asignación final sobre entrenamiento)
- `inertia_` (SSE)
Parámetros: `k`, `max_iter`, `tol`, `random_state`, `init` (random / k-means++ opcional), `n_init` (opcional).

---

## Consideraciones importantes de K-Means
- Criterio de parada: cambio máximo de centroides < `tol` o `max_iter`.
- Centroides vacíos: reubicar al punto más lejano o re-inicializar aleatoriamente.
- Normalización: ajustar en train y reutilizar para predict (guardar media/desvío).

---

## UI Streamlit (features mínimas)
- Selector de implementación: (loop / numpy / sklearn)
- Parámetros: `k`, `max_iter`, `tol`, `random_state`, `n_runs`
- Resultados:
	- Tabla/resumen de métricas por corrida
	- Gráfico comparativo para varios `k` (tiempo e inercia; silhouette opcional)
- Predicción de un registro nuevo:
	- Form con los 11 atributos numéricos
	- Mostrar cluster asignado y distancias a centroides (extra simple)

---

## Deploy
- Objetivo: subir a Streamlit Community Cloud.
- Necesario: `requirements.txt` + app ejecutable desde `app.py`.

---

## TO DO (próximo paso)
1) Implementar evaluación/comparativa (multi-k, multi-run, métricas y gráficos).
2) Armar UI Streamlit con selector único.
3) Dejar todo comentado (docstrings + type hints) y agregar instrucciones de ejecución.
4) Empaquetar para deploy en Streamlit Community Cloud + guía para informe PDF.

---

DONE
- Lectura del enunciado (readme.md) y extracción de requisitos.
- Scaffold del proyecto y dependencias.
- Loader ARFF + normalización z-score.
- Implementaciones K-Means (loop, numpy) y wrapper sklearn.
- Tests unitarios iniciales OK.

---

## Testing ejecutado (2026-01-08)
- `pytest` (5 tests):
	- `tests/test_data.py`: z-score y smoke load del ARFF.
	- `tests/test_kmeans.py`: convergencia y centroides esperados en toy data; inercia consistente loop vs numpy; manejo de clusters vacíos.
Resultado: **todos los tests pasaron**.

---

## Plan de implementación (detalle)
- Fase 1: Scaffold del proyecto (carpetas, `requirements.txt`, configuraciones básicas).
- Fase 2: Carga y preprocesado
	- Loader `winequality.arff` (scipy.io.arff o liac-arff como fallback).
	- Separar features/label, manejo de missing (si aparece), casteo a float32.
	- Normalización z-score (almacenar media/desvío para usar en predict y UI).
- Fase 3: Algoritmos
	- `kmeans_loop`: versión no vectorizada con mismo API, re-inicialización si hay centroides vacíos, criterio tol/max_iter.
	- `kmeans_numpy`: vectorizado usando broadcasting, misma semántica.
	- `kmeans_sklearn`: wrapper para comparativa (solo baseline, no usado en las propias).
- Fase 4: Evaluación/comparativa
	- Runner para múltiples `k` y `n_runs`, midiendo tiempo e inercia (y silhouette opcional si es viable).
	- Gráficos/tablas reutilizables por la UI (evitar duplicar lógica).
- Fase 5: UI Streamlit
	- Selector de implementación (loop/numpy/sklearn), controles para `k`, `max_iter`, `tol`, `n_runs`, `random_state`.
	- Visualizaciones: métricas por corrida, gráfico comparativo multi-k, distancias a centroides.
	- Formulario para predecir un nuevo registro (11 atributos) y mostrar asignación y distancias.
- Fase 6: Empaquetado y deploy
	- README de ejecución local y link a Streamlit Community Cloud.
	- requirements.txt alineado a versiones (streamlit, numpy, pandas, scipy, scikit-learn, matplotlib/plotly, pytest).

---

## Plan de testing
- Herramienta: `pytest`.
- Cobertura mínima por módulo:
	- `data.py`: carga del ARFF, tamaño esperado, normalización con media ~0 y std ~1, reproducibilidad del scaler.
	- `kmeans_loop` y `kmeans_numpy`: convergencia en datasets sintéticos pequeños, coincidencia de inercia entre ambas (tolerancia), manejo de centroides vacíos, determinismo con `random_state`.
	- `kmeans_sklearn`: wrapper devuelve forma y métricas esperadas.
	- `evaluation.py`: múltiples corridas devuelven estructura coherente y métricas no negativas.
- Tests rápidos (unitarios) sin Streamlit; la UI se validará manualmente + chequeo de que `app.py` importe sin romper.
- Datos de prueba: usar pequeños arrays sintéticos embebidos en los tests para no depender del ARFF completo en cada test (solo algunos tests de integración pueden cargar el ARFF para smoke test).

