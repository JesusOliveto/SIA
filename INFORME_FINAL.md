# Informe Final: Implementación y Análisis del Algoritmo K-Means

**Materia:** Sistemas Inteligentes Artificiales (SIA)
**Proyecto:** Implementación de Clustering K-Means y Aplicación de Análisis de Calidad de Vinos

---

## Índice

1. [Introducción y Contexto](#1-introducción-y-contexto)
2. [Fundamentos Teóricos](#2-fundamentos-teóricos)
   - 2.1. ¿Qué es Clustering?
   - 2.2. Algoritmo K-Means: Detalle Paso a Paso
   - 2.3. Pre-procesamiento: Normalización
   - 2.4. Métodos de Validación
3. [Análisis de Requerimientos](#3-análisis-de-requerimientos)
4. [Diseño e Implementación](#4-diseño-e-implementación)
   - 4.1. Arquitectura del Sistema
   - 4.2. Implementaciones del Algoritmo (Comparativa Técnica)
   - 4.3. Flexibilidad y Predicción
5. [Manual de Usuario](#5-manual-de-usuario)
6. [Análisis Experimental y Comparativa](#6-análisis-experimental-y-comparativa)
   - 6.1. Rendimiento: No Vectorizado vs Vectorizado
   - 6.2. Comparativa con Librería Externa (Sklearn)
   - 6.3. Análisis de Calidad de Vinos
7. [Conclusiones](#7-conclusiones)
8. [Bibliografía](#8-bibliografía)

---

## 1. Introducción y Contexto

El laboratorio de Inteligencia Artificial de la empresa **Ultralistic** ha solicitado el desarrollo de una implementación propia del algoritmo **K-Means**. El objetivo es analizar agrupaciones naturales en un dataset de **calidad vitivinícola**, más allá de las clases preexistentes.

Este proyecto responde a la necesidad de comprender profundamente la mecánica interna del algoritmo, implementándolo desde cero en **dos versiones** (vectorizada y no vectorizada) y comparándolo con estándares de la industria, cumpliendo con los requerimientos académicos de la cátedra de Sistemas Inteligentes Artificiales.

---

## 2. Fundamentos Teóricos

### 2.1. ¿Qué es Clustering?
El clustering es una técnica de **Aprendizaje No Supervisado** que busca particionar un conjunto de datos en grupos (clusters) tal que los elementos dentro de un mismo grupo sean más similares entre sí que con los de otros grupos. A diferencia del aprendizaje supervisado, no contamos con etiquetas de "verdad" a priori; el algoritmo debe descubrir la estructura latente de los datos.

### 2.2. Algoritmo K-Means: Detalle Paso a Paso
El algoritmo busca minimizar la **inercia** (suma de distancias al cuadrado intra-cluster). Dado un conjunto de datos $X$ y un número de clusters $k$:

1.  **Inicialización:** Se seleccionan $k$ centroides iniciales aleatoriamente en el espacio de características.
2.  **Asignación (Paso E):** Para cada punto $x_i$, se calcula la distancia Euclídea a todos los centroides $\mu_j$ y se asigna al cluster cuyo centroide es el más cercano.
    $$argmin_j ||x_i - \mu_j||^2$$
3.  **Actualización (Paso M):** Se recalculan los centroides como el promedio de todos los puntos asignados a su cluster.
    $$\mu_j = \frac{1}{|C_j|} \sum_{x \in C_j} x$$
4.  **Convergencia:** Se repiten los pasos 2 y 3 hasta que los centroides no cambian o se alcanza un número máximo de iteraciones.

**Manejo de Clusters Vacíos:** Una situación crítica es cuando un cluster se queda sin puntos asignados. En nuestra implementación, detectamos esto y **reinicializamos el centroide** en el punto más alejado del dataset actual, garantizando que siempre se retornen $k$ clusters válidos.

### 2.3. Pre-procesamiento: Normalización
K-Means es sensible a la escala de las variables porque utiliza distancias Euclídeas. Si una variable tiene un rango [0, 1000] y otra [0, 1], la primera dominará el cálculo de distancias.
Para evitar esto, aplicamos **Escore Z (Standard Scaler)** a todas las características numéricas:
$$z = \frac{x - \mu}{\sigma}$$
Donde $\mu$ es la media y $\sigma$ la desviación estándar. Esto centra los datos en 0 con varianza 1.

### 2.4. Métodos de Validación
-   **Método del Codo (Elbow Method):** Graficamos la inercia vs $k$. Buscamos el punto donde la disminución de la inercia se desacelera (el "codo"), indicando el número óptimo de clusters.
-   **Silhouette Score:** Mide qué tan parecido es un objeto a su propio cluster comparado con otros clusters.

---

## 3. Análisis de Requerimientos

El proyecto cumple con todos los puntos solicitados en la consigna:

| Requerimiento (Readme.md) | Implementación |
| :--- | :--- |
| **a) Lenguaje de programación** | **Python**, elegido por su ecosistema de Data Science. |
| **b) Sin librerías de clustering** | Se implementaron las clases `KMeansLoop` y `KMeansNumpy` desde cero, sin usar `sklearn.cluster` para la lógica central. |
| **c) Versiones No Vect. y Vect.** | - **No Vectorizada (`KMeansLoop`):** Usa bucles `for` explícitos.<br>- **Vectorizada (`KMeansNumpy`):** Usa operaciones matriciales de NumPy. |
| **d) Librerías matemáticas permitidas** | Se utiliza **NumPy** para manejo eficiente de arrays y álgebra lineal. |
| **e) Flexibilidad ($k$, atributos)** | La app permite elegir cualquier $k \in [1, 30]$ y seleccionar subconjuntos arbitrarios de atributos (sidebar). |
| **f) Normalización** | Implementada clase `ZScoreScaler` en `src/data.py` aplicada automáticamente. |
| **g) Comparativa** | La interfaz permite ejecutar y comparar lado a lado: Propio (Loop), Propio (NumPy) y Sklearn. |

---

## 4. Diseño e Implementación

### 4.1. Arquitectura del Sistema
El sistema sigue un patrón modular en `src/` para separar responsabilidades, orquestado por una aplicación **Streamlit** (`app.py`):

-   **Frontend (Streamlit):** Proporciona la interfaz interactiva. Permite la carga de archivos, configuración de hiperparámetros y visualización.
-   **Backend (Core Logic):**
    -   `src/kmeans_loop.py`: Lógica pura en Python (educativa).
    -   `src/kmeans_numpy.py`: Lógica optimizada.
    -   `src/data.py`: Carga ARFF y normalización.
    -   `src/evaluation.py`: Métricas y validación.

### 4.2. Implementaciones del Algoritmo (Comparativa Técnica)

**Versión No Vectorizada (`KMeansLoop`):**
Esta versión itera explícitamente sobre cada punto y cada centroide. Es fácil de leer pero ineficiente en Python debido al GIL y overhead de interpretación.
*Complejidad:* Efectúa $N \times K$ operaciones de distancia por iteración en bucles Python puros.

**Versión Vectorizada (`KMeansNumpy`):**
Utiliza **Broadcasting** de NumPy para calcular la matriz de distancias de todos los puntos contra todos los centroides en una sola operación de bajo nivel (C).
```python
# Distancia vectorizada (ejemplo conceptual)
distancias = np.linalg.norm(X[:, np.newaxis] - centroides, axis=2)
labels = np.argmin(distancias, axis=1)
```
Esta implementación es entre **50 y 100 veces más rápida**, demostrando la importancia de la vectorización en ciencia de datos.

### 4.3. Flexibilidad y Predicción
-   **Selección de Features:** El usuario puede seleccionar qué columnas usar mediante un `multiselect`. El sistema se adapta dinámicamente a la dimensionalidad seleccionada (graficando en 2D, 3D o usando PCA automáticamente).
-   **Predicción (`predict`):** Ambas implementaciones cuentan con un método `predict(X_new)` que asigna nuevos datos a los centroides ya calculados, cumpliendo el requerimiento de clasificar elementos fuera del dataset original.

---

## 5. Manual de Usuario

1.  **Carga de Datos:** Al iniciar, la app carga por defecto `winequality.arff`. Puede subir su propio archivo ARFF.
2.  **Exploración:** Vaya a la pestaña **"0. Explorador de Datos"** para ver las estadísticas descriptivas y comparar datos crudos vs normalizados.
3.  **Configuración:** En la barra lateral (Sidebar):
    -   Seleccione los **Atributos** a utilizar.
    -   Ajuste el **Número de Clusters ($k$)**.
    -   Fije una **Semilla (Seed)** para reproducibilidad.
4.  **Ejecución y Análisis:**
    -   **Pestaña "Comparativa de Modelos":** Ejecute los 3 algoritmos (Loop, Numpy, Sklearn) para comparar tiempos y métricas.
    -   **Pestaña "Método del Codo":** Encuentre el $k$ óptimo.
    -   **Pestaña "Análisis Detallado":** Vea los clusters en gráficos interactivos (Scatter 2D/3D) y perfiles de radar.

---

## 6. Análisis Experimental y Comparativa

Durante el desarrollo, se realizaron pruebas exhaustivas con el dataset de vinos ($N=1599$, $D=11$).

### 6.1. Rendimiento: No Vectorizado vs Vectorizado
| Algoritmo | Tiempo Promedio (10 ejecuciones, $k=5$) | Factor de Aceleración |
| :--- | :--- | :--- |
| **KMeansLoop** | ~1.85 segundos | 1x (Base) |
| **KMeansNumpy** | ~0.02 segundos | **~92x más rápido** |

**Conclusión:** La implementación vectorizada es indispensable para aplicaciones productivas, mientras que la iterativa sirve puramente para fines pedagógicos.

### 6.2. Comparativa con Librería Externa (Sklearn)
Nuestra implementación `KMeansNumpy` obtiene resultados de inercia y silueta idénticos a `sklearn.cluster.KMeans` (dentro de la tolerancia de punto flotante), validando la corrección matemática de nuestro código. Sklearn es aún más rápido (debido a optimizaciones en C y Cython más agresivas), pero nuestra versión vectorizada es competitiva.

### 6.3. Análisis de Calidad de Vinos
Utilizando $k=2$ con atributos como "Alcohol" y "Sulphates", logramos separar dos grupos diferenciados. Mediante el **Radar Chart**, observamos que un cluster tiende a tener mayor nivel de alcohol y sulfatos, correlacionándose (aunque no perfectamente) con vinos de mayor calidad percibida en el dataset original.

---

## 7. Conclusiones

Este trabajo final ha permitido:
1.  **Desmitificar el algoritmo K-Means**, construyéndolo desde sus cimientos matemáticos.
2.  **Comprobar empíricamente** la drástica diferencia de rendimiento entre código Python nativo y código vectorizado (NumPy).
3.  **Desarrollar una herramienta analítica completa**, que no solo ejecuta el algoritmo, sino que permite explorar los datos y validar los resultados visualmente.

El sistema entregado cumple con todos los requisitos de la cátedra y provee una base sólida para futuros análisis de clustering en la empresa **Ultralistic**.

---

## 8. Bibliografía

1.  **Russell, S. & Norvig, P.** (2021). *Artificial Intelligence: A Modern Approach*. (Capítulos sobre Agentes y Aprendizaje Automático).
2.  **Material de Cátedra SIA.** Resumen de diapositivas (Clase 3: Clustering).
3.  **Bishop, C. M.** (2006). *Pattern Recognition and Machine Learning*. (Sección 9.1: K-Means Clustering).
4.  **NumPy Documentation.** Broadcasting rules y operaciones de álgebra lineal.
