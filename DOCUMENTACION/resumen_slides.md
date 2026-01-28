# Resumen: Sistemas Inteligentes Artificiales

Este documento consolida los conceptos clave de las clases de Sistemas Inteligentes Artificiales, abarcando desde fundamentos de IA y agentes hasta algoritmos específicos de aprendizaje supervisado, no supervisado (con foco en K-Means) y evolutivos.

---

## 1. Introducción a la IA y Agentes (Clase 1)

### Definiciones de IA
[cite_start]Existen múltiples enfoques para definir la Inteligencia Artificial[cite: 1, 2]:
* **Procesos Mentales:** Sistemas que piensan como humanos ("máquinas con mentes").
* **Conducta:** Sistemas que actúan como humanos o racionalmente (realizan tareas que los humanos hacen mejor o estudian el diseño de agentes inteligentes).

### Agentes Inteligentes
[cite_start]Un agente es cualquier entidad capaz de percibir su entorno mediante **sensores** y actuar en él mediante **actuadores**[cite: 11].
* [cite_start]**Racionalidad:** Un agente racional busca maximizar su medida de rendimiento basándose en sus percepciones y conocimiento[cite: 20, 22].
* [cite_start]**Modelo REAS:** Para diseñar un agente se define: **R**endimiento, **E**ntorno, **A**ctuadores y **S**ensores[cite: 15].

**Tipos de Agentes:**
1.  [cite_start]**Reactivo Simple:** Actúa según la percepción actual (reglas condición-acción)[cite: 24].
2.  [cite_start]**Basado en Modelos:** Mantiene un estado interno sobre cómo evoluciona el mundo[cite: 24].
3.  [cite_start]**Basado en Objetivos:** Utiliza información sobre el objetivo deseado para decidir acciones[cite: 26].
4.  [cite_start]**Basado en Utilidad:** Evalúa qué tan "feliz" o eficiente es un estado (grado de satisfacción)[cite: 28].

---

## 2. Aprendizaje Automático: Conceptos y Árboles (Clase 2 y Cálculo de Información)

### Conceptos Generales
El aprendizaje automático (Machine Learning) es la capacidad de mejorar el rendimiento en una tarea a través de la experiencia.
* **Tipos de Aprendizaje:**
    * **Supervisado:** Se entrena con datos etiquetados (entradas y salidas conocidas). [cite_start]Ej: Clasificación (Árboles), Regresión[cite: 283].
    * **No Supervisado:** Datos no etiquetados (solo entradas). Busca estructuras ocultas. [cite_start]Ej: Clustering (K-Means)[cite: 280, 281].

### Árboles de Decisión (Supervisado)
Método de clasificación donde:
* **Nodos internos:** Representan una prueba sobre un atributo.
* **Ramas:** Representan los posibles valores del atributo.
* [cite_start]**Hojas:** Representan la clase final o decisión[cite: 287, 288].

**Construcción del Árbol (Algoritmo ID3):**
Se utiliza la **Ganancia de Información** basada en la **Entropía** para decidir qué atributo divide mejor los datos en cada paso.
* **Entropía:** Medida de incertidumbre o desorden.
* [cite_start]**Ganancia:** Reducción de la entropía al dividir por un atributo específico[cite: 3559, 3563].

---

## 3. Clustering y Algoritmo K-Means (Clase 3)
*Nota: Esta sección está expandida para tu implementación.*

### Introducción al Clustering
[cite_start]Técnica de aprendizaje **no supervisado** que agrupa objetos en conjuntos (clusters) de tal forma que los objetos del mismo grupo sean muy similares entre sí y diferentes a los de otros grupos[cite: 628, 630]. [cite_start]Requiere pre-procesamiento como la **normalización** (escalar datos a rangos [0,1] o similar) para que las distancias no se vean sesgadas por la magnitud de los atributos[cite: 660, 663].

### Algoritmo K-Means (Detalle de Implementación)
El objetivo es dividir $n$ observaciones en $k$ grupos, minimizando la varianza intra-cluster (la suma de las distancias cuadráticas entre los datos y el centroide de su grupo).

#### Estructura de Datos
* **Entrada:** Conjunto de datos $X$ (vectores numéricos de $d$ dimensiones) y el número de clusters $k$.
* **Salida:** $k$ centroides y la asignación de cada punto a un cluster.

#### Paso a Paso del Algoritmo
1.  **Inicialización (Semillas):**
    * Se seleccionan $k$ puntos iniciales como centroides (semillas). [cite_start]Pueden ser puntos aleatorios del dataset o generados al azar en el espacio de datos[cite: 1180].
    
2.  **Asignación (Expectation Step):**
    * Para cada punto de datos del dataset, se calcula la distancia (usualmente **Euclidiana cuadrática**) hacia cada uno de los $k$ centroides.
    * [cite_start]El punto se asigna al cluster cuyo centroide esté más cerca[cite: 1084, 1195].
    * [cite_start]*Matemáticamente:* Se busca minimizar $||x_j - \mu_i||^2$ donde $x_j$ es el dato y $\mu_i$ el centroide[cite: 636].

3.  **Actualización (Maximization Step):**
    * Una vez que todos los puntos tienen un cluster asignado, se recalculan los centroides.
    * [cite_start]El nuevo centroide de un cluster es el **promedio (media aritmética)** de todos los vectores asignados a ese cluster[cite: 1213].
    * *Formula:* $\mu_i = \frac{1}{|S_i|} \sum_{x \in S_i} x$

4.  **Iteración y Convergencia:**
    * Se repiten los pasos 2 y 3.
    * [cite_start]**Condición de parada:** El algoritmo se detiene cuando los centroides ya no cambian de posición (o el cambio es inferior a un umbral $\epsilon$) o cuando la asignación de puntos a clusters no varía[cite: 634].

#### Determinación del K (Método del Codo)
Dado que $k$ es un parámetro de entrada, se usa el "Método del Codo" (Elbow Method) para elegir el óptimo:
1.  Ejecutar K-Means para un rango de valores de $k$ (ej. de 1 a 10).
2.  Calcular la **Suma de Errores Cuadráticos (SSE)** o distancia media intra-cluster para cada $k$.
3.  Graficar SSE vs $k$.
4.  [cite_start]Elegir el $k$ donde la curva forma un "codo" (el punto donde aumentar $k$ ya no reduce drásticamente el error)[cite: 636, 640].

---

## 4. Regresión Lineal (Clase 4)

Método supervisado para predecir una variable continua (dependiente) en función de variables independientes.
* [cite_start]**Modelo:** Hipótesis lineal $h_\theta(x) = \theta_0 + \theta_1 x$ (recta que mejor se ajusta)[cite: 1703, 1712].
* [cite_start]**Función de Coste (MSE):** Se busca minimizar el Error Cuadrático Medio entre la predicción y el valor real[cite: 1751].
* **Descenso por el Gradiente:** Algoritmo iterativo para encontrar los parámetros $\theta$ que minimizan el costo. [cite_start]Actualiza los pesos moviéndose en dirección opuesta a la pendiente (derivada) de la función de error[cite: 1752, 1582].
    * $\alpha$ (tasa de aprendizaje): Controla el tamaño del paso. [cite_start]Si es muy grande, puede divergir; si es muy pequeño, es lento[cite: 1583].

---

## 5. Redes Neuronales: Introducción y Hopfield (Clase 5)

### Fundamentos
Inspiradas en la neurona biológica (soma, axón, dendritas, sinapsis).
* [cite_start]**Neurona Artificial:** Recibe entradas ponderadas por **pesos** ($W$), aplica una **función de activación** y genera una salida[cite: 2174, 2176].

### Modelo de Hopfield
[cite_start]Red recurrente monocapa utilizada como **memoria asociativa**[cite: 2179].
* [cite_start]**Características:** Conexiones "todos con todos" (feed-back), pesos simétricos ($w_{ij} = w_{ji}$), sin auto-conexiones ($w_{ii}=0$)[cite: 2181, 2182].
* [cite_start]**Aprendizaje (Hebbiano):** Los pesos se calculan una sola vez (aprendizaje off-line) basándose en los patrones a memorizar[cite: 2191, 2192].
* [cite_start]**Funcionamiento:** Dada una entrada (incluso incompleta o con ruido), la red itera hasta converger a uno de los patrones memorizados (estado de mínima energía)[cite: 2198].

---

## 6. Perceptrón, Backpropagation y Kohonen (Clase 6)

### Perceptrón Simple
Unidad básica de clasificación binaria para problemas **linealmente separables**.
* [cite_start]**Aprendizaje:** Ajusta los pesos $W$ en función del error entre la salida obtenida y la deseada: $w_{nuevo} = w_{actual} + \alpha \cdot (error) \cdot entrada$[cite: 2463, 2720].
* [cite_start]**Limitación:** No puede resolver problemas no lineales como la compuerta XOR[cite: 2486].

### Redes Multicapa (Backpropagation)
[cite_start]Soluciona la limitación del perceptrón añadiendo **capas ocultas** y usando funciones de activación no lineales (sigmoide, tangente hiperbólica)[cite: 2487, 2498].
* **Algoritmo:**
    1.  **Feed-forward:** La señal viaja de la entrada a la salida.
    2.  **Cálculo de error:** Se compara la salida con el objetivo.
    3.  [cite_start]**Back-propagation:** El error se propaga hacia atrás, ajustando los pesos de la capa de salida y luego los de las capas ocultas[cite: 2503, 2506].

### Redes de Kohonen (SOM)
Redes **competitivas** de aprendizaje **no supervisado**.
* **Objetivo:** Mapear datos de entrada (n-dimensiones) a un mapa discreto (generalmente 2D) preservando la topología.
* [cite_start]**Competencia:** Las neuronas compiten por activarse ante una entrada; solo gana una (la que tenga el vector de pesos más parecido a la entrada, distancia Euclídea mínima)[cite: 2538, 2541].
* [cite_start]**Vecindad:** La neurona ganadora y sus vecinas actualizan sus pesos para parecerse más a la entrada ("el ganador se lleva todo", pero arrastra a los vecinos)[cite: 2549].

---

## 7. Algoritmos Genéticos (Clase 7)

[cite_start]Métodos de búsqueda y optimización inspirados en la **evolución biológica** (selección natural)[cite: 3192].

### Ciclo del Algoritmo Genético
1.  **Población Inicial:** Individuos (cromosomas) generados aleatoriamente.
2.  [cite_start]**Evaluación (Fitness):** Se asigna un puntaje a cada individuo según qué tan bien resuelve el problema[cite: 3189].
3.  [cite_start]**Selección:** Se eligen los individuos para reproducirse, favoreciendo a los más aptos (Métodos: Ruleta, Torneo, Ranking, Elitismo)[cite: 3205, 3216].
4.  [cite_start]**Cruce (Crossover):** Combina partes de dos padres para crear descendencia (Cruce en un punto, multipunto)[cite: 3209].
5.  [cite_start]**Mutación:** Altera aleatoriamente algún gen con baja probabilidad para mantener la diversidad y evitar mínimos locales[cite: 3230].
6.  **Reemplazo:** La nueva generación sustituye a la anterior. Se repite hasta cumplir la condición de parada.

---

## Anexo Técnico: Implementación de K-Means

Esta sección detalla la lógica de programación necesaria para implementar K-Means, basándose en los conceptos de la **Clase 3**.

### 1. Pre-procesamiento de Datos (Normalización)
Antes de ejecutar K-Means, es **crítico** normalizar los datos. Como K-Means utiliza distancias (Euclídeas), si una variable tiene una magnitud mucho mayor que otra (ej. "Salario" vs "Edad"), dominará el cálculo y sesgará el resultado.

**Fórmula Min-Max Scaling (Escalado):**
Para cada atributo $x$ de tus datos:
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
* Esto deja todos los valores en el rango $[0, 1]$.

### 2. Estructura del Algoritmo (Pseudocódigo)

A continuación, el flujo lógico para programar la función `k_means`:

**Entradas:**
* `X`: Matriz de datos de tamaño $(N \times D)$ donde $N$ es el número de muestras y $D$ la dimensión (atributos).
* `K`: Número de clusters deseados.
* `max_iter`: Número máximo de iteraciones (seguridad para evitar bucles infinitos).
* `epsilon`: Tolerancia mínima de convergencia (ej. 0.001).

**Variables:**
* `centroides`: Matriz $(K \times D)$.
* `asignaciones`: Vector de tamaño $N$ (guarda a qué cluster pertenece cada dato).

**Algoritmo:**

```text
FUNCION K_Means(X, K, max_iter, epsilon):
    
    1. INICIALIZACIÓN
       Seleccionar K puntos aleatorios de X como 'centroides' iniciales.
       
    2. BUCLE PRINCIPAL (Repetir hasta convergencia o max_iter)
       Para i desde 1 hasta max_iter:
       
           A. PASO DE ASIGNACIÓN (Expectation)
              Para cada punto 'x' en X:
                  - Calcular la distancia euclidiana entre 'x' y cada uno de los K centroides.
                    Distancia(a, b) = RaizCuadrada( Suma((a_d - b_d)^2) )
                  - Asignar 'x' al centroide con la menor distancia.
                  - Guardar índice en 'asignaciones'.
           
           B. PASO DE ACTUALIZACIÓN (Maximization)
              Guardar una copia de 'centroides' actuales en 'centroides_anteriores'.
              Para cada cluster k desde 1 hasta K:
                  - Encontrar todos los puntos asignados al cluster k.
                  - Si el cluster no tiene puntos (vacío):
                      Re-inicializar ese centroide aleatoriamente (opcional para evitar error).
                  - Sino:
                      Calcular el PROMEDIO (media) de esos puntos dimensión por dimensión.
                      Nuevo 'centroide[k]' = Promedio de los puntos.
           
           C. CHEQUEO DE CONVERGENCIA
              Calcular el cambio total = Suma de distancias entre 'centroides' y 'centroides_anteriores'.
              Si (cambio total < epsilon):
                  ROMPER ciclo (El algoritmo convergió).
                  
    3. RETORNO
       Devolver 'centroides' y 'asignaciones'.
FIN FUNCION