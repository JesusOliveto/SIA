from __future__ import annotations

import numpy as np

from .kmeans_base import calcular_inercia
from .utils import asegurar_generador


class KMeansLoop:
    """
    Implementación de K-Means utilizando bucles explícitos de Python (sin vectorizar).

    ¿Qué hace?:
    Agrupa un conjunto de datos en 'num_clusters' particiones distintas basándose en la
    similitud (distancia Euclidiana) entre las muestras.

    ¿Cómo lo hace?:
    Itera muestra por muestra y centroide por centroide, calculando distancias con
    bucles 'for' anidados estándar de Python. Es `O(N*K*D)` en Python puro.

    Finalidad:
    Su único propósito es educativo y de depuración. Sirve para visualizar claramente
    la lógica granular y paso a paso del algoritmo (asignación y recálculo) sin el
    nivel de abstracción que introduce la vectorización matricial de NumPy.
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
        """
        Devuelve las coordenadas de los centros de los clusters.
        
        Finalidad: Proveer acceso seguro de solo lectura a los centroides 
        una vez que el algoritmo ha finalizado su entrenamiento.
        """
        if self.centroides_ is None:
            raise RuntimeError("El modelo no ha sido ajustado (fitted).")
        return self.centroides_

    @property
    def etiquetas(self) -> np.ndarray:
        """Devuelve las etiquetas de cada punto."""
        if self.etiquetas_ is None:
            raise RuntimeError("El modelo no ha sido ajustado (fitted).")
        return self.etiquetas_

    def ajustar(self, datos: np.ndarray) -> "KMeansLoop":
        """
        Calcula el clustering K-means utilizando bucles explícitos.

        ¿Qué hace?:
        Ejecuta el ciclo de entrenamiento completo. Puede ejecutar el algoritmo múltiples 
        veces (num_inicios) con diferentes semillas iniciales.

        ¿Cómo lo hace?:
        Itera 'num_inicios' veces llamando a '_ejecutar_corrida'. Al final, retiene el modelo
        con la menor inercia encontrada de entre todas las ejecuciones para evitar mínimos locales.

        Finalidad:
        Encontrar los mejores centroides para un conjunto de datos dado y asignar
        las etiquetas óptimas a cada muestra, preparándolo para inferencias.

        Args:
            datos: Datos estructurados de forma (num_muestras, num_caracteristicas).

        Returns:
            self: El estimador ajustado.
        """
        datos_np = np.asarray(datos, dtype=np.float64)
        mejor_inercia = np.inf
        mejores_centroides = None
        mejores_etiquetas = None
        mejor_num_iter = 0

        generador_base = asegurar_generador(self.estado_aleatorio)
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
        Ejecuta una única iteración (corrida independiente) del algoritmo K-Means.
        
        ¿Qué hace?:
        Inicializa aleatoriamente los centroides muestreando puntos del dataset 
        y realiza el ciclo iterativo clásico de Lloyd: 
        1) Asignar muestras, 2) Recalcular centroides.
        
        ¿Cómo lo hace?:
        Repite el ciclo hasta que el desplazamiento ('desplazamiento') de los centroides 
        en una iteración sea menor que una tolerancia 'tolerancia', o se alcance 'max_iteraciones'. 
        Al usar bucles 'for', permite inyectar fácilmente código de logging (detallado).
        
        Finalidad:
        Llevar a cabo la lógica algorítmica iterativa fundamental para llegar
        a una convergencia desde un punto de partida (semilla) específico.
        
        Args:
            datos: Datos de entrada.
            semilla: Semilla aleatoria para reproducibilidad en esta ejecución.
            
        Returns:
            Tupla de (centroides, etiquetas, inercia, num_iter).
        """
        generador = asegurar_generador(semilla)
        num_muestras = datos.shape[0]
        indices = generador.choice(num_muestras, size=self.num_clusters, replace=False)
        centroides = datos[indices].copy()

        etiquetas = np.zeros(num_muestras, dtype=np.int32)
        for iteracion in range(self.max_iteraciones):
            etiquetas = self._asignar_etiquetas(datos, centroides)
            nuevos_centroides, vacios = self._recalcular_centroides(datos, etiquetas)
            if vacios:
                self._arreglar_clusters_vacios(datos, nuevos_centroides, etiquetas, generador)

            desplazamiento = float(np.max(np.linalg.norm(nuevos_centroides - centroides, axis=1)))
            
            if self.detallado:
                inercia_actual = calcular_inercia(datos, nuevos_centroides, etiquetas)
                print(f"[Loop] Iter {iteracion+1}: desplazamiento={desplazamiento:.6f}, inercia={inercia_actual:.4f}")

            centroides = nuevos_centroides
            if desplazamiento <= self.tolerancia:
                if self.detallado:
                    print(f"[Loop] Convergencia en iter {iteracion+1}")
                break

        inercia = calcular_inercia(datos, centroides, etiquetas)
        return centroides, etiquetas, inercia, iteracion + 1

    def _asignar_etiquetas(self, datos: np.ndarray, centroides: np.ndarray) -> np.ndarray:
        """
        Paso 1 del algoritmo de Lloyd: Asignación.
        Asigna cada muestra al centroide más cercano utilizando bucles explícitos.
        
        ¿Cómo lo hace?:
        Itera sobre todas las muestras de 'datos', y para cada muestra evalúa la distancia
        cuadrada frente a cada centroide para determinar y retornar el índice más próximo.
        
        Finalidad:
        Mostrar de la forma más rudimentaria, pero clara para la enseñanza de Python, 
        cómo se deciden los clústeres iterando elemento a elemento (O(N*K*D)).
        """
        num_muestras = datos.shape[0]
        etiquetas = np.zeros(num_muestras, dtype=np.int32)
        for i in range(num_muestras):
            mejor_etiqueta = 0
            mejor_distancia = np.inf
            for idx_c in range(self.num_clusters):
                distancia = float(np.sum((datos[i] - centroides[idx_c]) ** 2))
                if distancia < mejor_distancia:
                    mejor_distancia = distancia
                    mejor_etiqueta = idx_c
            etiquetas[i] = mejor_etiqueta
        return etiquetas

    def _recalcular_centroides(self, datos: np.ndarray, etiquetas: np.ndarray):
        """
        Paso 2 del algoritmo de Lloyd: Actualización.
        Recalcula los centroides como el promedio ponderado de los puntos asignados.
        
        ¿Cómo lo hace?: Suma secuencialmente mediante índices explícitos y luego
        divide el total acumulado entre el conteo de muestras de cada clúster.

        Finalidad: Desplazar los centroides geométricamente hacia el centro de masa de
        su partición respectiva para minimizar iterativamente la inercia del sistema.
        Devuelve información sobre clústeres no asignados (vacíos) para corregirlos.
        """
        centroides = np.zeros((self.num_clusters, datos.shape[1]), dtype=np.float64)
        conteos = np.zeros(self.num_clusters, dtype=np.int32)
        for idx, etiqueta in enumerate(etiquetas):
            centroides[etiqueta] += datos[idx]
            conteos[etiqueta] += 1

        clusters_vacios = []
        for idx_c in range(self.num_clusters):
            if conteos[idx_c] == 0:
                clusters_vacios.append(idx_c)
            else:
                centroides[idx_c] /= conteos[idx_c]
        return centroides, clusters_vacios

    def _arreglar_clusters_vacios(self, datos: np.ndarray, centroides: np.ndarray, etiquetas: np.ndarray, generador: np.random.Generator) -> None:
        """
        Maneja la anomalía matemática de clusters vacíos (sin muestras asignadas).

        ¿Cómo lo hace?:
        Encuentra el punto 'datos' que tiene la mayor distancia global con respecto a 
        su centroide actual, y reubica allí el centroide huérfano (vacío).
        
        Finalidad:
        Asegurar robustez empírica del algoritmo. Sin esta gestión la varianza inter-cluster
        caería sin justificación y un centroide dejaría de impactar la partición del espacio.
        """
        distancias = np.zeros(datos.shape[0], dtype=np.float64)
        for i in range(datos.shape[0]):
            idx_c = etiquetas[i]
            distancias[i] = float(np.sum((datos[i] - centroides[idx_c]) ** 2))
        indice_mas_lejano = int(np.argmax(distancias))
        vacios = np.setdiff1d(np.arange(self.num_clusters), np.unique(etiquetas))
        for idx_c in vacios:
            centroides[idx_c] = datos[indice_mas_lejano]
            etiquetas[indice_mas_lejano] = idx_c

    def predecir(self, datos: np.ndarray) -> np.ndarray:
        """
        Inferencia de cluster. Predice a qué partición pertenece cada nueva muestra.

        ¿Qué hace?:
        No altera el estado del modelo entrenado. Simplemente proyecta nuevos datos 
        y los clasifica usando el mapa de distancias a los centroides establecidos.

        Finalidad:
        Habilitar la puesta en producción del agrupador sobre streams de
        datos venideros tras haber validado el modelo K-Means actual.

        Args:
            datos: Nuevos datos a predecir.

        Returns:
            Etiquetas de cluster.
        """
        if self.centroides_ is None:
            raise RuntimeError("El modelo no ha sido ajustado (fitted).")
        datos_np = np.asarray(datos, dtype=np.float64)
        return self._asignar_etiquetas(datos_np, self.centroides_)

    def ajustar_predecir(self, datos: np.ndarray) -> np.ndarray:
        """Calcula los centros de los clusters y predice el índice del cluster para cada muestra."""
        return self.ajustar(datos).etiquetas_

