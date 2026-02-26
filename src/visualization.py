from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def plot_pca_2d(X: np.ndarray, labels: np.ndarray) -> go.Figure:
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
        X: Datos de entrada escalar (N, D).
        labels: Etiquetas de cluster para cada punto (N,).

    Returns:
        Objeto Figure de Plotly.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df["Cluster"] = labels.astype(str)
    
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


def plot_radar_centroids(centers: np.ndarray, feature_names: List[str]) -> go.Figure:
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
        centers: Centros de cluster (K, D).
        feature_names: Nombres de características de longitud D.

    Returns:
        Objeto Figure de Plotly.
    """
    fig = go.Figure()
    
    # Close the loop for radar chart
    categories = feature_names + [feature_names[0]]
    
    for i, center in enumerate(centers):
        values = list(center)
        values += [values[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f'Cluster {i}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[centers.min(), centers.max()]
            )
        ),
        showlegend=True,
        title="Perfil de Centroides (Radar Chart)",
        template="plotly_white"
    )
    return fig


def plot_scatter_2d(X: np.ndarray, labels: np.ndarray, feature_names: List[str]) -> go.Figure:
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
        X: Datos de entrada (N, 2).
        labels: Etiquetas de cluster para cada punto (N,).
        feature_names: Lista con los 2 nombres de features.

    Returns:
        Objeto Figure de Plotly.
    """
    df = pd.DataFrame(X, columns=feature_names)
    df["Cluster"] = labels.astype(str)

    fig = px.scatter(
        df,
        x=feature_names[0],
        y=feature_names[1],
        color="Cluster",
        title=f"Clusters 2D: {feature_names[0]} vs {feature_names[1]}",
        template="plotly_white",
        opacity=0.7,
        labels={feature_names[0]: feature_names[0], feature_names[1]: feature_names[1]},
    )
    fig.update_layout(legend_title_text="Cluster")
    return fig


def plot_scatter_3d(X: np.ndarray, labels: np.ndarray, feature_names: List[str]) -> go.Figure:
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
        X: Datos de entrada (N, 3).
        labels: Etiquetas de cluster para cada punto (N,).
        feature_names: Lista con los 3 nombres de features.

    Returns:
        Objeto Figure de Plotly.
    """
    df = pd.DataFrame(X, columns=feature_names)
    df["Cluster"] = labels.astype(str)

    fig = px.scatter_3d(
        df,
        x=feature_names[0],
        y=feature_names[1],
        z=feature_names[2],
        color="Cluster",
        title=f"Clusters 3D: {feature_names[0]}, {feature_names[1]}, {feature_names[2]}",
        template="plotly_white",
        opacity=0.7,
    )
    fig.update_layout(legend_title_text="Cluster")
    return fig
