from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from src.data import DataBundle, normalize_bundle, load_winequality
from src.evaluation import evaluate_models
from src.kmeans_loop import KMeansLoop
from src.kmeans_numpy import KMeansNumpy
from src.kmeans_sklearn import KMeansSklearn
from src.visualization import plot_pca_2d, plot_radar_centroids


st.set_page_config(page_title="K-Means SIA", layout="wide")
st.title("K-Means — Calidad de Vino (SIA)")


@st.cache_data
def load_data(filename: str = "whinequalityclean.arff") -> tuple[DataBundle, DataBundle]:
	data_path = Path(__file__).resolve().parent / "datasets" / filename
	bundle = load_winequality(data_path)
	norm_bundle, scaler = normalize_bundle(bundle)
	# guardamos scaler dentro del bundle retornado vía closure para reuso
	norm_bundle.scaler = scaler  # type: ignore[attr-defined]
	return bundle, norm_bundle


def build_impls() -> Dict[str, str]:
	return {
		"loop": "K-Means (loops)",
		"numpy": "K-Means (NumPy)",
		"sklearn": "K-Means (sklearn baseline)",
	}


def builder_factory(name: str):
	if name == "loop":
		return lambda k, seed, verbose=False: KMeansLoop(n_clusters=k, n_init=3, random_state=seed, verbose=verbose)
	if name == "numpy":
		return lambda k, seed, verbose=False: KMeansNumpy(n_clusters=k, n_init=3, random_state=seed, verbose=verbose)
	if name == "sklearn":
		return lambda k, seed, verbose=False: KMeansSklearn(n_clusters=k, n_init=10, random_state=seed)
	raise ValueError(f"Implementación no soportada: {name}")


def run_evaluation(norm_bundle: DataBundle, ks: List[int], impl_keys: List[str], n_runs: int, with_silhouette: bool):
	builders = {k: builder_factory(k) for k in impl_keys}
	# evaluate_models expects builder(k, seed). run_evaluation needs to adapt to new factory that accepts verbose.
	# We'll stick to non-verbose for bulk evaluation.
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
		rows.append(
			{
				"impl": r.impl,
				"k": r.k,
				"run": r.run,
				"inertia": r.inertia,
				"fit_time": r.fit_time,
				"n_iter": r.n_iter,
				"silhouette": r.silhouette,
			}
		)
	df = pd.DataFrame(rows)
	summary = (
		df.groupby(["impl", "k"])
		.agg({"inertia": "mean", "fit_time": "mean", "n_iter": "mean", "silhouette": "mean"})
		.reset_index()
	)
	return df, summary


def predict_single(impl: str, k: int, sample: np.ndarray, norm_bundle: DataBundle, random_state: int):
	model = builder_factory(impl)(k, random_state, verbose=False)
	model.fit(norm_bundle.X)
	labels = model.predict(sample)
	distances = np.linalg.norm(model.cluster_centers_ - sample, axis=1)
	return int(labels[0]), distances, model


st.sidebar.header("Datos")
dataset_map = {
	"whinequalityclean.arff": "Limpio (Sin Outliers)",
	"winequality.arff": "Original (Completo)"
}
selected_filename = st.sidebar.selectbox(
	"Seleccionar Dataset",
	options=list(dataset_map.keys()),
	format_func=lambda x: dataset_map[x],
	index=0
)

bundle, norm_bundle = load_data(selected_filename)
st.sidebar.info(f"Registros: {bundle.X.shape[0]} | Atributos: {bundle.X.shape[1]}")

st.sidebar.header("Algoritmos")
impls = build_impls()
selected_impls = st.sidebar.multiselect(
	"Comparativa General",
	options=list(impls.keys()),
	format_func=lambda k: impls[k],
	default=list(impls.keys()),
)

k_min, k_max = st.sidebar.slider("Rango de k (Comparativa)", 2, 12, (3, 8))
n_runs = st.sidebar.slider("Corridas por k", 1, 5, 2)
with_silhouette = st.sidebar.checkbox("Calcular silhouette", value=True)

st.markdown("---")
st.markdown("---")
st.header("1. Determinación de K óptimo (Método del Codo)")
st.caption("Esta sección ayuda a identificar la cantidad ideal de clusters ($k$). Busque el punto donde la reducción de la Inercia (error) se estabiliza, formando un 'codo' en el gráfico.")

col_codo1, col_codo2 = st.columns([1, 2])
with col_codo1:
    k_elbow_max = st.slider("K máximo para analizar", 5, 15, 10)
    trigger_elbow = st.button("Calcular Codo")

with col_codo2:
    if trigger_elbow:
        with st.spinner("Calculando inercia para rango de k..."):
            elbow_data = []
            for k_val in range(1, k_elbow_max + 1):
                # Use numpy impl for speed, seed 42 for stability
                model = builder_factory("numpy")(k_val, 42, verbose=False)
                model.fit(norm_bundle.X)
                elbow_data.append({"k": k_val, "Inercia (SSE)": model.inertia_})
            
            df_elbow = pd.DataFrame(elbow_data)
            st.line_chart(df_elbow, x="k", y="Inercia (SSE)")
            st.info("Observe dónde la curva empieza a aplanarse. Ese suele ser un buen valor para $k$.")

st.markdown("---")
st.header("2. Comparativa de Implementaciones")
st.caption("Ejecuta múltiples configuraciones en lote para comparar el rendimiento (Tiempo) y la calidad (Inercia, Silhouette) de las distintas implementaciones (Loops vs NumPy vs Sklearn).")

if st.button("Ejecutar comparativa"):
	if not selected_impls:
		st.warning("Selecciona al menos una implementación.")
	else:
		ks = list(range(k_min, k_max + 1))
		with st.spinner("Corriendo comparativas..."):
			df_runs, df_summary = run_evaluation(norm_bundle, ks, selected_impls, n_runs, with_silhouette)
		st.success("Listo")
		st.write("Resultados por corrida")
		st.dataframe(df_runs, use_container_width=True)

		st.write("Promedio por implementación y k")
		st.dataframe(df_summary, use_container_width=True)

		st.write("Inercia vs k (promedio)")
		inertia_chart = df_summary.pivot(index="k", columns="impl", values="inertia")
		st.line_chart(inertia_chart)

		st.write("Tiempo de ajuste vs k (s)")
		time_chart = df_summary.pivot(index="k", columns="impl", values="fit_time")
		st.line_chart(time_chart)

		if with_silhouette:
			st.write("Silhouette vs k")
			sil_chart = df_summary.pivot(index="k", columns="impl", values="silhouette")
			st.line_chart(sil_chart)


st.markdown("---")
st.markdown("---")
st.header("3. Predicción de un nuevo registro")
st.caption("Permite simular la llegada de un nuevo dato (vino) y clasificarlo en uno de los clusters existentes. Muestra el proceso interno de decisión.")

col1, col2 = st.columns(2)
with col1:
	impl_pred = st.selectbox("Implementación para predecir", list(impls.keys()), format_func=lambda k: impls[k])
	k_pred = st.slider("k para predicción", 2, 12, 4)
	seed_pred = st.number_input("random_state", min_value=0, value=42, step=1)

with col2:
	st.write("Completa los 11 atributos")
	feature_vals = []
	for name in bundle.feature_names:
		default_val = float(np.mean(bundle.X[:, bundle.feature_names.index(name)]))
		val = st.number_input(name, value=default_val)
		feature_vals.append(val)

if st.button("Predecir cluster"):
	sample = np.array(feature_vals, dtype=np.float64).reshape(1, -1)
	
	# Paso 1: Normalización
	st.markdown("### Paso 1: Normalización")
	st.info("Los datos se normalizan (Z-Score) para equiparar el peso de todas las variables.")
	
	scaler = getattr(norm_bundle, "scaler")
	sample_norm = scaler.transform(sample)  # type: ignore[assignment]
	
	col_n1, col_n2 = st.columns(2)
	with col_n1:
		st.write("Input Original:", pd.DataFrame(sample, columns=bundle.feature_names))
	with col_n2:
		st.write("Input Normalizado:", pd.DataFrame(sample_norm, columns=bundle.feature_names))

	# Ejecución
	label, distances, model = predict_single(impl_pred, k_pred, sample_norm, norm_bundle, int(seed_pred))

	# Paso 2: Distancias
	st.markdown("### Paso 2: Distancias a Centroides")
	st.info(f"Se calcula la distancia Euclídea contra los {k_pred} centroides obtenida por el modelo.")
	
	dist_df = pd.DataFrame({
		"Cluster": list(range(len(distances))),
		"Distancia": distances,
		"Seleccionado": ["SI" if i == label else "NO" for i in range(len(distances))]
	})
	st.dataframe(dist_df.style.highlight_min(subset=["Distancia"], color="lightgreen", axis=0), use_container_width=True)

	# Paso 3: Resultado
	st.markdown("### Paso 3: Decisión Final")
	st.success(f"Cluster Asignado: {label}")
	st.write(f"El punto se asigna al Cluster **{label}** porque es el que minimiza la distancia ({distances[label]:.4f}).")

st.markdown("---")
st.markdown("---")
st.markdown("---")
st.markdown("---")
st.header("4. Análisis y Ejecución Controlada")
st.caption("Profundice en una configuración específica. Use el **Modo Depuración** para ver paso a paso cómo converge el algoritmo, o el **Modo Rendimiento** para obtener estadísticas robustas y gráficos detallados.")

import contextlib
import io
import time
from src.kmeans_base import compute_inertia
from sklearn.metrics import silhouette_score

col_viz1, col_viz2 = st.columns([1, 2])
with col_viz1:
	viz_impl = st.selectbox("Algoritmo", ["loop", "numpy", "sklearn"], key="viz_impl", format_func=lambda x: impls[x])
	viz_k = st.slider("k", 2, 10, 4, key="viz_k")
	viz_mode = st.radio("Modo de Ejecución", ["Rendimiento", "Depuración"], key="viz_mode")
	viz_seed = st.number_input("Semilla", value=42, key="viz_seed")
	
	run_btn = st.button("Ejecutar Análisis")

with col_viz2:
	if run_btn:
		if viz_mode == "Depuración":
			st.markdown("### Salida de Depuración")
			log_capture = io.StringIO()
			
			with st.spinner("Ejecutando paso a paso..."):
				with contextlib.redirect_stdout(log_capture):
					# Debug run: verbose=True, n_init=1 (to avoid noise)
					start_t = time.perf_counter()
					# Factory signature is (k, seed, verbose)
					model = builder_factory(viz_impl)(viz_k, int(viz_seed), verbose=True)
					# Force n_init to 1 for clearer logs in debug
					if hasattr(model, 'n_init'):
						model.n_init = 1
						print(f"[Debug] Configured n_init=1 for transparency.")
					
					model.fit(norm_bundle.X)
					end_t = time.perf_counter()
					print(f"[Done] Fit time: {end_t - start_t:.4f}s")
			
			st.code(log_capture.getvalue(), language="text")
			
			st.success("Modelo Ajustado")
			
			# Metrics
			inertia = getattr(model, "inertia_", 0.0)
			sil = silhouette_score(norm_bundle.X, model.labels_) if len(set(model.labels_)) > 1 else -1
			
			m1, m2, m3 = st.columns(3)
			m1.metric("Tiempo", f"{end_t - start_t:.4f}s")
			m1.metric("Inercia (SSE)", f"{inertia:.2f}")
			m3.metric("Silhouette", f"{sil:.3f}")
			
			# Quick Visuals
			tab_pca, tab_radar = st.tabs(["Mapa PCA", "Radar Centroides"])
			with tab_pca:
				st.plotly_chart(plot_pca_2d(norm_bundle.X, model.labels_), use_container_width=True)
			with tab_radar:
				st.plotly_chart(plot_radar_centroids(model.cluster_centers_, bundle.feature_names), use_container_width=True)
				
		else: # Modo Rendimiento
			st.markdown("### Estadísticas de Rendimiento")
			N_Perf_Runs = 5
			with st.spinner(f"Ejecutando {N_Perf_Runs} corridas para promediar..."):
				inertias = []
				times = []
				sils = []
				
				best_model = None
				min_inertia = float('inf')
				
				for i in range(N_Perf_Runs):
					seed = int(viz_seed) + i
					t0 = time.perf_counter()
					model = builder_factory(viz_impl)(viz_k, seed, verbose=False)
					model.fit(norm_bundle.X)
					t1 = time.perf_counter()
					
					inertias.append(model.inertia_)
					times.append(t1 - t0)
					if len(set(model.labels_)) > 1:
						sils.append(silhouette_score(norm_bundle.X, model.labels_))
					else:
						sils.append(-1.0)
						
					if model.inertia_ < min_inertia:
						min_inertia = model.inertia_
						best_model = model
				
				# Display Stats
				df_stats = pd.DataFrame({
					"Métrica": ["Tiempo Medio", "Inercia Media", "Silhouette Medio"],
					"Valor": [np.mean(times), np.mean(inertias), np.mean(sils)],
					"Desvío": [np.std(times), np.std(inertias), np.std(sils)]
				})
				st.dataframe(df_stats)
				
				# Best Model Visuals
				st.write("#### Visualización del mejor modelo hallado")
				tab_pca, tab_radar = st.tabs(["Mapa PCA", "Radar Centroides"])
				with tab_pca:
					st.plotly_chart(plot_pca_2d(norm_bundle.X, best_model.labels_), use_container_width=True)
				with tab_radar:
					st.plotly_chart(plot_radar_centroids(best_model.cluster_centers_, bundle.feature_names), use_container_width=True)
