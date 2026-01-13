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


st.set_page_config(page_title="K-Means SIA", layout="wide")
st.title("K-Means — Calidad de Vino (SIA)")


@st.cache_data
def load_data() -> tuple[DataBundle, DataBundle]:
	data_path = Path(__file__).resolve().parent / "winequality.arff"
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
		return lambda k, seed: KMeansLoop(n_clusters=k, n_init=3, random_state=seed)
	if name == "numpy":
		return lambda k, seed: KMeansNumpy(n_clusters=k, n_init=3, random_state=seed)
	if name == "sklearn":
		return lambda k, seed: KMeansSklearn(n_clusters=k, n_init=10, random_state=seed)
	raise ValueError(f"Implementación no soportada: {name}")


def run_evaluation(norm_bundle: DataBundle, ks: List[int], impl_keys: List[str], n_runs: int, with_silhouette: bool):
	builders = {k: builder_factory(k) for k in impl_keys}
	results = evaluate_models(
		norm_bundle.X,
		ks=ks,
		builders=builders,
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
	model = builder_factory(impl)(k, random_state)
	model.fit(norm_bundle.X)
	labels = model.predict(sample)
	distances = np.linalg.norm(model.cluster_centers_ - sample, axis=1)
	return int(labels[0]), distances


bundle, norm_bundle = load_data()
st.sidebar.header("Configuración")
st.sidebar.write(f"Registros: {bundle.X.shape[0]} | Atributos: {bundle.X.shape[1]}")

impls = build_impls()
selected_impls = st.sidebar.multiselect(
	"Implementaciones",
	options=list(impls.keys()),
	format_func=lambda k: impls[k],
	default=list(impls.keys()),
)

k_min, k_max = st.sidebar.slider("Rango de k", 2, 12, (3, 8))
n_runs = st.sidebar.slider("Corridas por k", 1, 5, 2)
with_silhouette = st.sidebar.checkbox("Calcular silhouette", value=True)

st.markdown("---")
st.subheader("Comparativa de implementaciones")

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
st.subheader("Predicción de un nuevo registro")

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
	# usar scaler aprendido en normalización
	scaler = getattr(norm_bundle, "scaler")
	sample_norm = scaler.transform(sample)  # type: ignore[assignment]
	label, distances = predict_single(impl_pred, k_pred, sample_norm, norm_bundle, int(seed_pred))
	st.success(f"Cluster asignado: {label}")
	st.write("Distancias a centroides:")
	st.write({f"c{idx}": float(d) for idx, d in enumerate(distances)})
