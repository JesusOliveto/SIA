from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def graficar_pca_2d(datos: np.ndarray, etiquetas: np.ndarray) -> go.Figure:
    """
    Proyecta los datos de alta dimensionalidad a 2D utilizando PCA.
    
    ¿Qué hace?:
    Toma un dataset de `D` dimensiones (ej. 11 propiedades físico-químicas del vino) 
    y lo comprime en las 2 componentes ortogonales que capturan la mayor varianza.
    Luego lo grafica en un scatter plot coloreado por clúster.
    
    ¿Cómo lo hace?:
    Delega en `sklearn.decomposition.PCA` la transformación matemática (Extracción de
    eigenvectores de la matriz de covarianza) y usa Plotly Express para el mapeo interactivo 2D.
    
    Finalidad:
    Visualizar clústeres hiperdimensionales (`D > 3`) en una pantalla plana. Sirve
    para auditar visualmente el trabajo del agrupador: Si K-Means fue exitoso, a menudo 
    veremos manchas de color bien separadas en el plano de Componentes Principales.

    Args:
        datos: Datos de entrada escalar (N, D).
        etiquetas: Etiquetas de cluster para cada punto (N,).

    Returns:
        Objeto Figure de Plotly.
    """
    pca = PCA(n_components=2)
    datos_pca = pca.fit_transform(datos)
    
    df = pd.DataFrame(datos_pca, columns=["PC1", "PC2"])
    df["Cluster"] = etiquetas.astype(str)
    
    fig = px.scatter(
        df, 
        x="PC1", 
        y="PC2", 
        color="Cluster",
        title="Proyección PCA 2D de los Clusters",
        template="plotly_white",
        opacity=0.7
    )
    fig.update_layout(legend_title_text="Cluster")
    return fig


def graficar_radar_centroides(centroides: np.ndarray, nombres_caracteristicas: List[str]) -> go.Figure:
    """
    Genera un gráfico de radar (araña) multivariable comparando perfiles de centroides.
    
    ¿Qué hace?:
    Grafica una red poligonal donde cada eje radial es una característica (feature) del
    dataset y los vértices internos del polígono marcan el centro de masa del clúster
    en esa dimensión particular.
    
    ¿Cómo lo hace?:
    Extrae iterativamente las coordenadas `(K, D)` de cada centroide, "cierra el ciclo"
    vectorial duplicando la última coordenada hacia el origen, e insta a `Plotly Scatterpolar`
    a graficar el área de influencia de cada grupo superpuestos.
    
    Finalidad:
    Proveer "Explicabilidad" (Explainable AI) a los usuarios no técnicos. En vez de
    entregarles una etiqueta arbitraria '0,1,2', esto les permite perfilar semánticamente 
    a cada clúster (ej. "Vino Rico en Azúcar y Bajo en Alcohol").

    Args:
        centroides: Centros de cluster (K, D).
        nombres_caracteristicas: Nombres de características de longitud D.

    Returns:
        Objeto Figure de Plotly.
    """
    fig = go.Figure()
    
    # Close the loop for radar chart
    categorias = nombres_caracteristicas + [nombres_caracteristicas[0]]
    
    for i, centroide in enumerate(centroides):
        valores = list(centroide)
        valores += [valores[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=valores,
            theta=categorias,
            fill='toself',
            name=f'Cluster {i}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[centroides.min(), centroides.max()]
            )
        ),
        showlegend=True,
        title="Perfil de Centroides (Radar Chart)",
        template="plotly_white"
    )
    return fig


def graficar_dispersion_2d(datos: np.ndarray, etiquetas: np.ndarray, nombres_caracteristicas: List[str]) -> go.Figure:
    """
    Gráfico de dispersión bidimensional estándar.

    ¿Qué hace?: 
    Plotea muestras y sus clústeres asignados directamente sobre sus 2 variables nativas.
    
    ¿Cómo lo hace?:
    Evita cualquier álgebra lineal reductiva y plotea con Plotly Express directamente
    los datos pre-filtrados desde la UI.
    
    Finalidad:
    Permitir al usuario probar manualmente la hipótesis de agrupamiento para pares
    de características específicas que le resulten de interés (ej. pH vs Densidad).

    Args:
        datos: Datos de entrada (N, 2).
        etiquetas: Etiquetas de cluster para cada punto (N,).
        nombres_caracteristicas: Lista con los 2 nombres de features.

    Returns:
        Objeto Figure de Plotly.
    """
    df = pd.DataFrame(datos, columns=nombres_caracteristicas)
    df["Cluster"] = etiquetas.astype(str)

    fig = px.scatter(
        df,
        x=nombres_caracteristicas[0],
        y=nombres_caracteristicas[1],
        color="Cluster",
        title=f"Clusters 2D: {nombres_caracteristicas[0]} vs {nombres_caracteristicas[1]}",
        template="plotly_white",
        opacity=0.7,
        labels={nombres_caracteristicas[0]: nombres_caracteristicas[0], nombres_caracteristicas[1]: nombres_caracteristicas[1]},
    )
    fig.update_layout(legend_title_text="Cluster")
    return fig


def graficar_dispersion_3d(datos: np.ndarray, etiquetas: np.ndarray, nombres_caracteristicas: List[str]) -> go.Figure:
    """
    Gráfico de dispersión tridimensional topológico.

    ¿Qué hace?:
    Aprovecha WebGL en el navegador para renderizar y rotar una nube de puntos
    codificada por colores de clústeres para 3 predictores numéricos simultáneos.
    
    Finalidad:
    Ayudar a visualizar la verdadera separación Euclidiana espacial de K-Means, 
    la cual es su asunción geométrica intrínseca primaria (Clusters esféricos, 
    no separaciones complicadas concéntricas). 

    Args:
        datos: Datos de entrada (N, 3).
        etiquetas: Etiquetas de cluster para cada punto (N,).
        nombres_caracteristicas: Lista con los 3 nombres de features.

    Returns:
        Objeto Figure de Plotly.
    """
    df = pd.DataFrame(datos, columns=nombres_caracteristicas)
    df["Cluster"] = etiquetas.astype(str)

    fig = px.scatter_3d(
        df,
        x=nombres_caracteristicas[0],
        y=nombres_caracteristicas[1],
        z=nombres_caracteristicas[2],
        color="Cluster",
        title=f"Clusters 3D: {nombres_caracteristicas[0]}, {nombres_caracteristicas[1]}, {nombres_caracteristicas[2]}",
        template="plotly_white",
        opacity=0.7,
    )
    fig.update_layout(legend_title_text="Cluster")
    return fig
