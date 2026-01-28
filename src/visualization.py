from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def plot_pca_2d(X: np.ndarray, labels: np.ndarray) -> go.Figure:
    """
    Project data to 2D using PCA and plot scatter with cluster colors.
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
    Plot a radar chart comparing centroid values for each feature.
    Assumes centers are already in a reasonable scale (e.g. normalized or original).
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
