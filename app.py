"""
Streamlit Application for K-Means Clustering Analysis.
Course: Sistemas Inteligentes Artificiales (SIA)
Project: K-Means Implementation for Wine Quality Analysis
"""
from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import silhouette_score

from src.data import DataBundle, normalize_bundle, load_winequality, ZScoreScaler
from src.evaluation import evaluate_models
from src.kmeans_loop import KMeansLoop
from src.kmeans_numpy import KMeansNumpy
from src.kmeans_sklearn import KMeansSklearn
from src.visualization import plot_pca_2d, plot_radar_centroids, plot_scatter_2d, plot_scatter_3d


st.set_page_config(
    page_title="SIA - Analisis K-Means",
    layout="wide",
    page_icon=None
)

# --- Header & Sidebar ---
st.title("Sistema de Análisis de Calidad Vitivinícola")
st.markdown("**Departamento de Inteligencia Artificial - Ultralistic**")
st.markdown("---")

st.sidebar.title("Configuración")
st.sidebar.markdown("**Panel de Control**")

@st.cache_data
def load_data(filename: str = "whinequalityclean.arff") -> Tuple[DataBundle, DataBundle]:
    """
    Loads and normalizes the dataset.
    
    Args:
        filename: Name of the ARFF file in datasets folder.
        
    Returns:
        Tuple of (Raw Bundle, Normalized Bundle).
    """
    data_path = Path(__file__).resolve().parent / "datasets" / filename
    bundle = load_winequality(data_path)
    norm_bundle, scaler = normalize_bundle(bundle)
    # Store scaler for later reuse (e.g. in predictions)
    norm_bundle.scaler = scaler  # type: ignore[attr-defined]
    return bundle, norm_bundle


def build_impls() -> Dict[str, str]:
    """Returns the available K-Means implementations mapping."""
    return {
        "loop": "K-Means (Python Loops)",
        "numpy": "K-Means (NumPy Vectorized)",
        "sklearn": "K-Means (Scikit-Learn Standard)",
    }


def builder_factory(name: str):
    """
    Factory to create KMeans instances.
    
    Args:
        name: Implementation key ('loop', 'numpy', 'sklearn').
        
    Returns:
        A callable that returns a fitted model.
    """
    if name == "loop":
        return lambda k, seed, verbose=False: KMeansLoop(n_clusters=k, n_init=3, random_state=seed, verbose=verbose)
    if name == "numpy":
        return lambda k, seed, verbose=False: KMeansNumpy(n_clusters=k, n_init=3, random_state=seed, verbose=verbose)
    if name == "sklearn":
        return lambda k, seed, verbose=False: KMeansSklearn(n_clusters=k, n_init=10, random_state=seed)
    raise ValueError(f"Unsupported implementation: {name}")


def run_evaluation(norm_bundle: DataBundle, ks: List[int], impl_keys: List[str], n_runs: int, with_silhouette: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs evaluating benchmarks across multiple K values and implementations.
    """
    # Adapt factory to match evaluate_models signature (k, seed) -> model
    adapted_builders = {
        k: lambda n_k, s: builder_factory(k)(n_k, s, verbose=False) 
        for k in impl_keys
    }
    
    results = evaluate_models(
        norm_bundle.X,
        ks=ks,
        builders=adapted_builders,
        n_runs=n_runs,
        random_state=42,
        compute_silhouette=with_silhouette,
        y_true=norm_bundle.y
    )
    
    rows = []
    for r in results:
        rows.append({
            "Implementación": r.impl,
            "k": r.k,
            "Run": r.run,
            "Inercia": r.inertia,
            "Tiempo (s)": r.fit_time,
            "Iteraciones": r.n_iter,
            "Silhouette": r.silhouette,
            "ARI": r.ari,
            "NMI": r.nmi
        })
        
    df = pd.DataFrame(rows)
    
    # Summary statistics
    # Filter out None columns mainly for ARI/NMI if strictly not present (but they should be None if not computed)
    agg_dict = {"Inercia": "mean", "Tiempo (s)": "mean", "Iteraciones": "mean", "Silhouette": "mean"}
    if norm_bundle.y is not None:
        agg_dict["ARI"] = "mean"
        agg_dict["NMI"] = "mean"
        
    summary = (
        df.groupby(["Implementación", "k"])
        .agg(agg_dict)
        .reset_index()
    )
    return df, summary


def predict_single(impl: str, k: int, sample: np.ndarray, norm_bundle: DataBundle, random_state: int):
    """
    Predicts the cluster for a single sample.
    
    IMPORTANT: This function retrains the model from scratch every time it is called,
    as per requirements to demonstrate the algorithm's behavior without caching.
    """
    # 1. Instantiate and Train (Fresh)
    model = builder_factory(impl)(k, random_state, verbose=False)
    model.fit(norm_bundle.X)
    
    # 2. Predict
    labels = model.predict(sample)
    distances = np.linalg.norm(model.cluster_centers_ - sample, axis=1)
    
    return int(labels[0]), distances, model


# --- Sidebar Inputs ---
st.sidebar.subheader("Selección de Datos")
dataset_map = {
    "whinequalityclean.arff": "Dataset Limpio (Recomendado)",
    "winequality.arff": "Dataset Original (Con Outliers)"
}
selected_filename = st.sidebar.selectbox(
    "Archivo de Origen",
    options=list(dataset_map.keys()),
    format_func=lambda x: dataset_map[x],
    index=0
)

# Load Data
try:
    with st.spinner("Cargando dataset..."):
        bundle, norm_bundle = load_data(selected_filename)
    st.sidebar.success(f"Cargado: {len(bundle.X)} registros, {len(bundle.feature_names)} atributos")
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()


st.sidebar.subheader("Parámetros Globales")
impls = build_impls()
selected_impls = st.sidebar.multiselect(
    "Implementaciones a Comparar",
    options=list(impls.keys()),
    format_func=lambda k: impls[k],
    default=list(impls.keys()),
)

k_range = st.sidebar.slider("Rango de Clusters (k)", 2, 12, (3, 8))
n_runs = st.sidebar.slider("Corridas por Configuración", 1, 5, 2, help="Promediar resultados para mayor robustez.")
with_silhouette = st.sidebar.checkbox("Calcular Silhouette Score", value=True)

st.sidebar.subheader("Selección de Atributos")
selected_features = st.sidebar.multiselect(
    "Atributos a utilizar en el algoritmo",
    options=bundle.feature_names,
    default=bundle.feature_names,
    help="Seleccione al menos 2 atributos. Con 2 o 3 se graficará directamente; con más se usará PCA."
)

if len(selected_features) < 2:
    st.sidebar.error("⚠️ Seleccione al menos 2 atributos.")

# Compute filtered data based on selected features
feature_indices = [bundle.feature_names.index(f) for f in selected_features]
X_selected = norm_bundle.X[:, feature_indices] if feature_indices else norm_bundle.X
feature_names_selected = selected_features if selected_features else bundle.feature_names


# --- Tabs for Main Sections ---
tab_explore, tab_elbow, tab_compare, tab_predict, tab_debug = st.tabs([
    "0. Explorador de Datos",
    "1. Método del Codo",
    "2. Comparativa",
    "3. Predicción",
    "4. Análisis Detallado"
])

# --- 0. Data Explorer ---
with tab_explore:
    st.header("Explorador de Datos")
    st.markdown("Inspeccione los datos antes de ejecutar el algoritmo. Cambie entre la vista cruda y normalizada para verificar el pre-procesamiento.")

    view_mode = st.radio("Vista", ["Datos Crudos", "Datos Normalizados"], horizontal=True)

    if view_mode == "Datos Crudos":
        df_view = pd.DataFrame(bundle.X, columns=bundle.feature_names)
        st.caption(f"Mostrando datos originales — {len(df_view)} registros, {len(bundle.feature_names)} atributos")
    else:
        df_view = pd.DataFrame(norm_bundle.X, columns=bundle.feature_names)
        st.caption(f"Mostrando datos normalizados (Z-Score) — {len(df_view)} registros, {len(bundle.feature_names)} atributos")

    st.dataframe(df_view, use_container_width=True, height=280)

    col_stats1, col_stats2 = st.columns([2, 1])

    with col_stats1:
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(
            df_view.describe().T.style.format("{:.4f}"),
            use_container_width=True
        )

    with col_stats2:
        st.subheader("Atributo Individual")
        selected_attr = st.selectbox("Seleccionar atributo", options=bundle.feature_names)
        if selected_attr:
            col_data = df_view[selected_attr]
            st.metric("Nombre", selected_attr)
            st.metric("Mínimo", f"{col_data.min():.5f}")
            st.metric("Máximo", f"{col_data.max():.5f}")
            st.metric("Media", f"{col_data.mean():.5f}")
            st.metric("Desv. Estándar", f"{col_data.std():.5f}")

# --- 1. Elbow Method ---
with tab_elbow:
    st.header("Determinación de K óptimo")
    st.info("Utilice el **Método del Codo** para identificar el número óptimo de clusters. Busque el punto donde la ganancia de inercia disminuye drásticamente.")
    
    col_e1, col_e2 = st.columns([1, 3])
    with col_e1:
        k_elbow_max = st.number_input("K Máximo", min_value=5, max_value=50, value=15)
        trigger_elbow = st.button("Generar Gráfico", type="primary")
        
    with col_e2:
        if trigger_elbow:
            if len(selected_features) < 2:
                st.warning("Seleccione al menos 2 atributos en el sidebar para continuar.")
            else:
              with st.spinner("Calculando curva de inercia..."):
                elbow_data = []
                # Always use NumPy for speed here
                factory = builder_factory("numpy")

                progress_bar = st.progress(0)
                for i, k_val in enumerate(range(1, k_elbow_max + 1)):
                    model = factory(k_val, 42, verbose=False)
                    model.fit(X_selected)
                    elbow_data.append({"k": k_val, "Inercia": model.inertia_})
                    progress_bar.progress((i + 1) / k_elbow_max)
                
                df_elbow = pd.DataFrame(elbow_data)
                
                fig_elbow = px.line(
                    df_elbow, 
                    x="k", 
                    y="Inercia", 
                    title="Análisis del Codo (Inercia vs k)",
                    markers=True,
                    labels={"k": "Número de Clusters (k)", "Inercia": "Suma de Errores al Cuadrado (SSE)"}
                )
                fig_elbow.update_layout(xaxis=dict(dtick=1))
                st.plotly_chart(fig_elbow, use_container_width=True)

# --- 2. Comparison ---
with tab_compare:
    st.header("Comparativa de Rendimiento y Calidad")
    st.markdown("Ejecute múltiples implementaciones para validar que los resultados sean consistentes y comparar tiempos de ejecución.")
    
    if st.button("Iniciar Comparativa", key="btn_compare"):
        if not selected_impls:
            st.warning("Por favor seleccione al menos una implementación en la barra lateral.")
        else:
            if len(selected_features) < 2:
                st.warning("Seleccione al menos 2 atributos en el sidebar para continuar.")
            else:
              ks = list(range(k_range[0], k_range[1] + 1))
              # Use a temporary norm_bundle slice for evaluation
              import dataclasses
              norm_bundle_sel = dataclasses.replace(norm_bundle, X=X_selected, feature_names=feature_names_selected)
              with st.spinner("Ejecutando benchmarks..."):
                df_runs, df_summary = run_evaluation(norm_bundle_sel, ks, selected_impls, n_runs, with_silhouette)
            
            st.success("Evaluación completada con éxito.")
            
            st.subheader("Resumen General")
            st.markdown(
                """
                **Interpretación de Métricas:**
                - **Inercia (SSE):** Cohesión interna (Menor es mejor).
                - **Silhouette:** Definición de clusters (-1 a 1, Mayor es mejor).
                - **ARI (Adjusted Rand Index):** Coincidencia con la calidad real del vino (0 a 1, Mayor es mejor). Mide la utilidad real.
                - **Tiempo:** Eficiencia computacional.
                """
            )

            # Format config
            fmt_dict = {
                "Inercia": "{:.2f}", 
                "Tiempo (s)": "{:.6f}", 
                "Silhouette": "{:.3f}",
                "Iteraciones": "{:.1f}"
            }
            if "ARI" in df_summary.columns:
                fmt_dict["ARI"] = "{:.3f}"
            if "NMI" in df_summary.columns:
                fmt_dict["NMI"] = "{:.3f}"

            # Highlighting
            st.dataframe(
                df_summary.style.format(fmt_dict)
                .background_gradient(subset=["Tiempo (s)"], cmap="RdYlGn_r")
                .background_gradient(subset=["Inercia"], cmap="Blues_r")
                .highlight_max(subset=["Silhouette"], color="#d4edda"),
                use_container_width=True
            )
            
            # Graphs
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Tiempo de Ejecución vs K")
                st.line_chart(df_summary.pivot(index="k", columns="Implementación", values="Tiempo (s)"))
                
                if "numpy" in selected_impls and "loop" in selected_impls:
                    # Speedup Calculation
                    np_times = df_summary[df_summary["Implementación"] == "numpy"].set_index("k")["Tiempo (s)"]
                    loop_times = df_summary[df_summary["Implementación"] == "loop"].set_index("k")["Tiempo (s)"]
                    speedup = loop_times / np_times
                    
                    st.markdown("#### Speedup (Loop / Numpy)")
                    st.bar_chart(speedup)
                    st.caption("Factor de aceleración: Cuántas veces más rápido es NumPy vs Python Loops.")

            with c2:
                if "ARI" in df_summary.columns:
                    st.markdown("#### Validación Externa (ARI) vs K")
                    st.line_chart(df_summary.pivot(index="k", columns="Implementación", values="ARI"))
                    st.caption("¿Los clusters coinciden con la calidad del vino? (ARI > 0 indica correlación)")
                else:
                    st.markdown("#### Inercia vs K")
                    st.line_chart(df_summary.pivot(index="k", columns="Implementación", values="Inercia"))

# --- 3. Prediction ---
with tab_predict:
    st.header("Simulación de Predicción")
    st.markdown("Simule la llegada de una nueva muestra de vino. El sistema **re-entrenará el modelo** con todo el dataset y clasificará la muestra.")
    
    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        st.subheader("Configuración del Modelo")
        p_impl = st.selectbox("Algoritmo", list(impls.keys()), format_func=lambda k: impls[k])
        p_k = st.slider("Número de Clusters (k)", 2, 12, 4, key="pred_k")
        p_seed = st.number_input("Semilla Aleatoria", value=42, step=1)
        
        btn_predict = st.button("Clasificar Muestra", type="primary")
        
    with col_p2:
        st.subheader("Atributos de la Muestra")
        st.caption("Modifique los valores para definir el nuevo vino.")
        
        input_vals = []
        cols = st.columns(3)
        for i, name in enumerate(bundle.feature_names):
            # Default to mean values for convenience
            default_val = float(np.mean(bundle.X[:, i]))
            with cols[i % 3]:
                val = st.number_input(name, value=default_val, format="%.4f")
                input_vals.append(val)
                
    if btn_predict:
        if len(selected_features) < 2:
            st.warning("Seleccione al menos 2 atributos en el sidebar para continuar.")
        else:
          sample = np.array(input_vals, dtype=np.float64).reshape(1, -1)

          # Normalize
          scaler = getattr(norm_bundle, "scaler")
          sample_norm = scaler.transform(sample)  # type: ignore[assignment]
          sample_norm = sample_norm[:, feature_indices]
        
          import dataclasses
          norm_bundle_pred = dataclasses.replace(norm_bundle, X=X_selected, feature_names=feature_names_selected)

          with st.status("Procesando...", expanded=True) as status:
              st.write("1. Normalizando datos de entrada...")
              time.sleep(0.3) # UX pause

              st.write(f"2. Entrenando modelo {impls[p_impl]} con k={p_k} desde cero...")
              start_t = time.perf_counter()
              label, distances, model = predict_single(p_impl, p_k, sample_norm, norm_bundle_pred, int(p_seed))
              end_t = time.perf_counter()
              st.write(f"Ref: Modelo entrenado en {end_t - start_t:.4f}s")

              st.write("3. Calculando distancias a centroides finales...")
              status.update(label="Clasificación Completada", state="complete", expanded=False)

          # Results
          col_res1, col_res2 = st.columns([1, 1])
          with col_res1:
              st.metric("Cluster Asignado", f"Cluster {label}")
              st.info(f"El vino ha sido clasificado en el Grupo {label}.")

          with col_res2:
              st.markdown(f"**Distancia mínima:** {distances[label]:.4f}")

          # Distances Table
          dist_df = pd.DataFrame({
              "Cluster ID": range(p_k),
              "Distancia Euclidiana": distances,
              "Estado": ["ASIGNADO" if i == label else "-" for i in range(p_k)]
          })
          st.table(dist_df.style.highlight_min(subset=["Distancia Euclidiana"], color="#d4edda", axis=0))

# --- 4. Deep Analysis ---
with tab_debug:
    st.header("Interpretación y Depuración")
    
    col_d1, col_d2 = st.columns([1, 2])
    
    with col_d1:
        st.markdown("##### Configuración")
        d_impl = st.selectbox("Algoritmo", ["loop", "numpy", "sklearn"], key="d_impl", format_func=lambda x: impls[x])
        d_k = st.slider("k", 2, 10, 4, key="d_k")
        d_seed = st.number_input("Semilla", value=42, key="d_seed")
        
        st.markdown("##### Modo")
        d_mode = st.radio("Tipo de Análisis", ["Perfilado (Heatmap)", "Depuración (Logs)"])
        d_run = st.button("Ejecutar Análisis")
        
    with col_d2:
        if d_run:
            if len(selected_features) < 2:
                st.warning("Seleccione al menos 2 atributos en el sidebar para continuar.")
            elif d_mode == "Depuración (Logs)":
                st.markdown("#### Logs de Ejecución Paso a Paso")
                log_capture = io.StringIO()

                with st.spinner("Ejecutando con logging activado..."):
                    with contextlib.redirect_stdout(log_capture):
                        model = builder_factory(d_impl)(d_k, int(d_seed), verbose=True)
                        if hasattr(model, 'n_init'):
                            model.n_init = 1

                        model.fit(X_selected)

                st.text_area("Traza del Algoritmo", log_capture.getvalue(), height=400)

            else: # Stats/Profiling Mode
                st.markdown("#### Perfilado de Clusters (Heatmap)")
                with st.spinner("Generando mapa de calor..."):
                    model = builder_factory(d_impl)(d_k, int(d_seed), verbose=False)
                    model.fit(X_selected)

                    # Create Heatmap Data using selected features
                    centers = model.cluster_centers_
                    feat_names = feature_names_selected

                    df_centers = pd.DataFrame(centers, columns=feat_names)
                    df_centers.index.name = "Cluster"

                    fig_heat = px.imshow(
                        df_centers,
                        labels=dict(x="Característica", y="Cluster", color="Z-Score"),
                        x=feat_names,
                        y=[f"Cluster {i}" for i in range(d_k)],
                        color_continuous_scale="RdBu_r",
                        aspect="auto",
                        title="Mapa de Calor de Centroides (Desviación Estándar)"
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                    st.caption("Colores Rojos indican valores por encima del promedio. Azules indican por debajo.")

                st.markdown("#### Visualización Espacial")
                n_sel = len(feature_names_selected)
                if n_sel == 2:
                    tab_labels = ["Scatter 2D Directo", "Radar Chart"]
                elif n_sel == 3:
                    tab_labels = ["Scatter 3D Directo", "Radar Chart"]
                else:
                    tab_labels = ["PCA 2D", "Radar Chart"]

                t_scatter, t_radar = st.tabs(tab_labels)
                with t_scatter:
                    if n_sel == 2:
                        st.plotly_chart(plot_scatter_2d(X_selected, model.labels_, feature_names_selected), use_container_width=True)
                        st.caption("Gráfico directo: los ejes representan los 2 atributos seleccionados (sin reducción de dimensionalidad).")
                    elif n_sel == 3:
                        st.plotly_chart(plot_scatter_3d(X_selected, model.labels_, feature_names_selected), use_container_width=True)
                        st.caption("Gráfico directo 3D interactivo: los ejes representan los 3 atributos seleccionados.")
                    else:
                        st.plotly_chart(plot_pca_2d(X_selected, model.labels_), use_container_width=True)
                        st.caption(f"Proyección PCA 2D aplicada sobre los {n_sel} atributos seleccionados.")
                with t_radar:
                    st.plotly_chart(plot_radar_centroids(model.cluster_centers_, feature_names_selected), use_container_width=True)
