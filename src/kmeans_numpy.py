from __future__ import annotations

import numpy as np

from .kmeans_base import calcular_inercia
from .utils import asegurar_generador


class KMeansNumpy:
    """
    Implementación de K-Means utilizando vectorización integral con NumPy.

    ¿Qué hace?:
    Agrupa un conjunto de datos particionándolo en 'num_clusters' utilizando la 
    metáfora probabilística del algoritmo iterativo original.

    ¿Cómo lo hace?:
    Reempleza los costosos iteradores `for` anidados de Python por tensores 
    y manipulación de ejes de memoria lineal usando "broadcasting". 

    Finalidad:
    Demostrar el impacto monumental del cálculo matricial delegando las instrucciones
    a C por debajo de la interfaz NumPy. Muestra un "punto intermedio" que es 
    suficientemente rápido para la vida real pero al que el desarrollador aún
    puede acceder para entender la implementación.
    """

    def __init__(self, num_clusters: int, max_iteraciones: int = 300, tolerancia: float = 1e-4, num_inicios: int = 1, estado_aleatorio: int | None = None, detallado: bool = False) -> None:
        if num_clusters <= 0:
            raise ValueError("num_clusters debe ser positivo.")
        self.num_clusters = num_clusters
        self.max_iteraciones = max_iteraciones
        self.tolerancia = tolerancia
        self.num_inicios = num_inicios
        self.estado_aleatorio = estado_aleatorio
        self.detallado = detallado

        self.centroides_: np.ndarray | None = None
        self.etiquetas_: np.ndarray | None = None
        self.inercia_: float | None = None
        self.num_iteraciones_: int | None = None

    @property
    def centroides(self) -> np.ndarray:
        if self.centroides_ is None:
            raise RuntimeError("El modelo no ha sido ajustado (fitted).")
        return self.centroides_

    @property
    def etiquetas(self) -> np.ndarray:
        if self.etiquetas_ is None:
            raise RuntimeError("El modelo no ha sido ajustado (fitted).")
        return self.etiquetas_

    def ajustar(self, datos: np.ndarray) -> "KMeansNumpy":
        """
        Calcula el clustering K-means utilizando operaciones tensor-completas de NumPy.

        ¿Qué hace?:
        Coordina las iteraciones y los arranques ('num_inicios') del algoritmo buscando
        el mejor modelo posible (aquél con la métrica de inercia más baja).

        ¿Cómo lo hace?:
        Controla los reinicios y se apoya en `_ejecutar_corrida` para gestionar la convergencia
        vectorizada internamente. Garantiza conversiones estrictas a float64
        para evitar overflows de puntero flotante.
        
        Finalidad:
        Intermediario principal de la API para recibir un dataset completo, entrenar el
        modelo reteniendo las etiquetas/centroides idóneos, y devolver la propia instancia 
        ajustada (encadenamiento de métodos).
        """
        datos_np = np.asarray(datos, dtype=np.float64)
        generador_base = asegurar_generador(self.estado_aleatorio)

        mejor_inercia = np.inf
        mejores_centroides = None
        mejores_etiquetas = None
        mejor_num_iter = 0

        for _ in range(self.num_inicios):
            semilla = int(generador_base.integers(0, 1_000_000_000))
            centroides, etiquetas, inercia, num_iter = self._ejecutar_corrida(datos_np, semilla)
            if inercia < mejor_inercia:
                mejor_inercia = inercia
                mejores_centroides = centroides
                mejores_etiquetas = etiquetas
                mejor_num_iter = num_iter

        self.centroides_ = np.asarray(mejores_centroides, dtype=np.float64)
        self.etiquetas_ = np.asarray(mejores_etiquetas, dtype=np.int32)
        self.inercia_ = float(mejor_inercia)
        self.num_iteraciones_ = int(mejor_num_iter)
        return self

    def _ejecutar_corrida(self, datos: np.ndarray, semilla: int):
        """
        Ejecuta una única corrida (ejecución independiente) del algoritmo K-Means.
        
        ¿Qué hace?:
        Lleva a cabo el proceso de inicialización y las fases de Lloyd: 
        1) Actualizar Asignaciones, 2) Actualizar Centroides.
        
        ¿Cómo lo hace?:
        Vectoriza brutalmente tanto la evaluación euclidiana como la recalculación.
        La condición de parada se computa determinando la norma matricial del 
        desplazamiento de clústeres (np.linalg.norm(nuevos_centroides - centroides, axis=1)).
        
        Finalidad:
        Motor de procesamiento principal. Busca la convergencia en el menor tiempo 
        de cómputo posible evitando traspasos de memoria Python-runtime perjudiciales.
        """
        generador = asegurar_generador(semilla)
        num_muestras = datos.shape[0]
        indices = generador.choice(num_muestras, size=self.num_clusters, replace=False)
        centroides = datos[indices].copy()

        etiquetas = np.zeros(num_muestras, dtype=np.int32)
        for iteracion in range(self.max_iteraciones):
            distancias = self._distancias_pares(datos, centroides)
            etiquetas = np.argmin(distancias, axis=1)

            nuevos_centroides, vacios = self._recalcular_centroides(datos, etiquetas)
            if vacios:
                self._arreglar_clusters_vacios(datos, nuevos_centroides, distancias, etiquetas)

            desplazamiento = float(np.max(np.linalg.norm(nuevos_centroides - centroides, axis=1)))
            
            if self.detallado:
                print(f"[Numpy] Iter {iteracion+1}: desplazamiento={desplazamiento:.6f}")

            centroides = nuevos_centroides
            if desplazamiento <= self.tolerancia:
                if self.detallado:
                    print(f"[Numpy] Convergencia en iter {iteracion+1}")
                break

        inercia = calcular_inercia(datos, centroides, etiquetas)
        return centroides, etiquetas, inercia, iteracion + 1

    def _distancias_pares(self, datos: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        """
        Motor geométrico: Calcula distancias utilizando 'broadcasting'.
        
        ¿Qué hace?:
        Computa la distancia entre toda pareja de `(muestra, centroide)` posible 
        escalando dinámicamente las dimensiones de los tensores.
        
        ¿Cómo lo hace?:
        Proyecta 'datos' de formato (N, D) a (N, 1, D) y altera 'centroides' a (1, K, D). 
        A partir de allí la resta se 'expande', generando un tensor 3D de 
        formato (N, K, D), para luego hacer sum() cuadrática sobre el eje D.
        
        Finalidad:
        Garantizar el punto más diferencial (O(1) ciclos for en Python) entre el 
        algoritmo base y la optimización NumPy. Devuelve la malla
        distancial necesaria para que argmin identifique los clusters.
        """
        diferencia = datos[:, None, :] - centroides[None, :, :]
        return np.sum(diferencia * diferencia, axis=2)

    def _recalcular_centroides(self, datos: np.ndarray, etiquetas: np.ndarray):
        """
        Actualiza los centroides utilizando indexación booleana de NumPy.
        
        ¿Cómo lo hace?:
        Recuenta asignaciones con `np.bincount` de antemano. Itera sólo a través de los
        `K` clústeres para aplicar un filtrado booleano eficiente del tipo `datos[etiquetas == idx_c]`
        calculando su media de forma nativa en C con `.mean(axis=0)`.

        Finalidad:
        Mover cada clúster de forma ultra-rápida y concurrente a su verdadero centro de masa 
        sin el rastreo iterativo secuencial. Retorna además si un cluster se quedó sin datos.
        """
        centroides = np.zeros((self.num_clusters, datos.shape[1]), dtype=np.float64)
        conteos = np.bincount(etiquetas, minlength=self.num_clusters).astype(np.int32)
        for idx_c in range(self.num_clusters):
            if conteos[idx_c] > 0:
                centroides[idx_c] = datos[etiquetas == idx_c].mean(axis=0)
        vacios = [c for c in range(self.num_clusters) if conteos[c] == 0]
        return centroides, vacios

    def _arreglar_clusters_vacios(self, datos: np.ndarray, centroides: np.ndarray, distancias: np.ndarray, etiquetas: np.ndarray) -> None:
        """
        Mapea el algoritmo de salvaguarda contra "Singularidad" en operaciones tensoriales.
        
        ¿Cómo lo hace?:
        Usa indexaciones avanzadas como `distancias[np.arange(...), etiquetas]` para 
        acceder de forma O(1) a la matriz de desviaciones, buscando sistemáticamente
        el mayor outlier de todos a relocalizar y "salvar" el clúster inerte.

        Finalidad:
        Prevenir la formación de sub-clústeres vacíos donde `.mean(...)` retornaría
        Valores nulos, lo que corrompe la convergencia geométrica de la vectorización en curso.
        """
        mas_cercanos = distancias[np.arange(distancias.shape[0]), etiquetas]
        indice_mas_lejano = int(np.argmax(mas_cercanos))
        vacios = np.setdiff1d(np.arange(self.num_clusters), np.unique(etiquetas))
        for idx_c in vacios:
            centroides[idx_c] = datos[indice_mas_lejano]
            etiquetas[indice_mas_lejano] = idx_c

    def predecir(self, datos: np.ndarray) -> np.ndarray:
        if self.centroides_ is None:
            raise RuntimeError("El modelo no ha sido ajustado (fitted).")
        datos_np = np.asarray(datos, dtype=np.float64)
        distancias = self._distancias_pares(datos_np, self.centroides_)
        return np.argmin(distancias, axis=1)

    def ajustar_predecir(self, datos: np.ndarray) -> np.ndarray:
        return self.ajustar(datos).etiquetas_
