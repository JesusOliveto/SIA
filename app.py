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
from src.visualization import plot_pca_2d, plot_radar_centroids


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
        })
        
    df = pd.DataFrame(rows)
    
    # Summary statistics
    summary = (
        df.groupby(["Implementación", "k"])
        .agg({"Inercia": "mean", "Tiempo (s)": "mean", "Iteraciones": "mean", "Silhouette": "mean"})
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


# --- Tabs for Main Sections ---
tab_elbow, tab_compare, tab_predict, tab_debug = st.tabs([
    "1. Método del Codo", 
    "2. Comparativa", 
    "3. Predicción", 
    "4. Análisis Detallado"
])

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
            with st.spinner("Calculando curva de inercia..."):
                elbow_data = []
                # Always use NumPy for speed here
                factory = builder_factory("numpy")
                
                progress_bar = st.progress(0)
                for i, k_val in enumerate(range(1, k_elbow_max + 1)):
                    model = factory(k_val, 42, verbose=False)
                    model.fit(norm_bundle.X)
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
            ks = list(range(k_range[0], k_range[1] + 1))
            with st.spinner("Ejecutando benchmarks..."):
                df_runs, df_summary = run_evaluation(norm_bundle, ks, selected_impls, n_runs, with_silhouette)
            
            st.success("Evaluación completada con éxito.")
            
            st.subheader("Resumen General")
            st.markdown(
                """
                **Interpretación de Métricas:**
                - **Inercia (SSE):** Mide la cohesión interna de los clusters. Valores más bajos indican clusters más compactos.
                - **Silhouette:** Mide qué tan bien definidos están los clusters (-1 a 1). Valores cercanos a 1 son mejores.
                - **Tiempo:** Tiempo de entrenamiento en segundos. Muestra la eficiencia del algoritmo.
                """
            )

            # Highlighting and Formatting
            st.dataframe(
                df_summary.style.format({
                    "Inercia": "{:.2f}", 
                    "Tiempo (s)": "{:.6f}", 
                    "Silhouette": "{:.3f}",
                    "Iteraciones": "{:.1f}"
                }).background_gradient(subset=["Tiempo (s)"], cmap="RdYlGn_r")  # Red for slow, Green for fast
                  .background_gradient(subset=["Inercia"], cmap="Blues_r")
                  .highlight_max(subset=["Silhouette"], color="#d4edda"),
                use_container_width=True
            )
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("#### Inercia (Menos es mejor)")
                st.line_chart(df_summary.pivot(index="k", columns="Implementación", values="Inercia"))
            with col_c2:
                st.markdown("#### Tiempo de Ejecución (Menos es mejor)")
                st.line_chart(df_summary.pivot(index="k", columns="Implementación", values="Tiempo (s)"))
                
            if with_silhouette:
                st.markdown("#### Silhouette Score (Más es mejor, -1 a 1)")
                st.line_chart(df_summary.pivot(index="k", columns="Implementación", values="Silhouette"))

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
        sample = np.array(input_vals, dtype=np.float64).reshape(1, -1)
        
        # Normalize
        scaler = getattr(norm_bundle, "scaler")
        sample_norm = scaler.transform(sample)  # type: ignore[assignment]
        
        with st.status("Procesando...", expanded=True) as status:
            st.write("1. Normalizando datos de entrada...")
            time.sleep(0.3) # UX pause
            
            st.write(f"2. Entrenando modelo {impls[p_impl]} con k={p_k} desde cero...")
            start_t = time.perf_counter()
            label, distances, model = predict_single(p_impl, p_k, sample_norm, norm_bundle, int(p_seed))
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
    st.header("Análisis y Depuración")
    st.markdown("Ejecución paso a paso para visualizar la convergencia del algoritmo.")
    
    col_d1, col_d2 = st.columns([1, 2])
    
    with col_d1:
        d_impl = st.selectbox("Algoritmo", ["loop", "numpy", "sklearn"], key="d_impl", format_func=lambda x: impls[x])
        d_k = st.slider("k", 2, 10, 4, key="d_k")
        d_mode = st.radio("Modo", ["Depuración (Logs)", "Rendimiento (Stats)"])
        d_seed = st.number_input("Semilla", value=42, key="d_seed")
        d_run = st.button("Ejecutar")
        
    with col_d2:
        if d_run:
            if d_mode == "Depuración (Logs)":
                st.markdown("#### Logs de Ejecución")
                log_capture = io.StringIO()
                
                with st.spinner("Ejecutando..."):
                    with contextlib.redirect_stdout(log_capture):
                        # Force verbose=True
                        model = builder_factory(d_impl)(d_k, int(d_seed), verbose=True)
                        # Quick hack to force 1 run for cleaner logs if supported
                        if hasattr(model, 'n_init'):
                            model.n_init = 1
                            print("[Debug] n_init forzado a 1 para limpieza de logs.")
                        
                        t0 = time.perf_counter()
                        model.fit(norm_bundle.X)
                        t1 = time.perf_counter()
                        print(f"\n[Finalizado] Tiempo total: {t1 - t0:.4f}s")
                
                st.text_area("Salida del Algoritmo", log_capture.getvalue(), height=300)
                
                # Visualizations
                st.subheader("Visualización del Resultado")
                t_pca, t_radar = st.tabs(["PCA 2D", "Radar"])
                with t_pca:
                    st.plotly_chart(plot_pca_2d(norm_bundle.X, model.labels_), use_container_width=True)
                with t_radar:
                    st.plotly_chart(plot_radar_centroids(model.cluster_centers_, bundle.feature_names), use_container_width=True)
                    
            else: # Stats Mode
                st.markdown("#### Estadísticas de Estabilidad")
                N_STATS = 5
                inertias, times, sils = [], [], []
                
                progress = st.progress(0)
                for i in range(N_STATS):
                    s = int(d_seed) + i
                    t0 = time.perf_counter()
                    m = builder_factory(d_impl)(d_k, s, verbose=False)
                    m.fit(norm_bundle.X)
                    t1 = time.perf_counter()
                    
                    inertias.append(m.inertia_)
                    times.append(t1 - t0)
                    if len(set(m.labels_)) > 1:
                        sils.append(silhouette_score(norm_bundle.X, m.labels_))
                    else:
                        sils.append(-1)
                    progress.progress((i+1)/N_STATS)
                
                st.success("Completado.")
                
                met_df = pd.DataFrame({
                    "Métrica": ["Tiempo", "Inercia", "Silhouette"],
                    "Promedio": [np.mean(times), np.mean(inertias), np.mean(sils)],
                    "Desviación Std": [np.std(times), np.std(inertias), np.std(sils)]
                })
                st.dataframe(met_df)
