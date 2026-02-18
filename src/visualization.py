from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def plot_pca_2d(X: np.ndarray, labels: np.ndarray) -> go.Figure:
    """
    Proyecta los datos a 2D utilizando PCA y genera un gráfico de dispersión con el color de los clusters.
    
    ¿Por qué PCA?
    -------------
    Como los datos tienen múltiples dimensiones (`D > 3`), no podemos graficarlos directamente.
    PCA (Análisis de Componentes Principales) reduce la dimensionalidad conservando la mayor
    varianza posible, permitiéndonos ver la separación de los clusters en un plano 2D.

    Args:
        X: Datos de entrada (N, D).
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
    Genera un gráfico de radar (araña) comparando los valores de los centroides para cada característica.
    
    ¿Por qué Radar Chart?
    ---------------------
    Permite visualizar el "perfil" promedio de cada cluster. Es ideal para distinguir cualitativamente
    qué características predominan en cada grupo (ej. un cluster con "Alta Acidez" vs uno con "Alto Alcohol").

    Args:
        centers: Centros de cluster (K, D).
        feature_names: Lista de nombres de características de longitud D.

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
    Genera un scatter plot 2D directo cuando se seleccionan exactamente 2 features.

    A diferencia de plot_pca_2d, este gráfico no aplica ninguna reducción de dimensionalidad,
    por lo que los ejes representan directamente los atributos seleccionados por el usuario.

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
    Genera un scatter plot 3D interactivo directo cuando se seleccionan exactamente 3 features.

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
