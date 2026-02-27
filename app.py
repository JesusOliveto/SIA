"""
Aplicación Streamlit para Análisis de Agrupamiento K-Means.
Curso: Sistemas Inteligentes Artificiales (SIA)
Proyecto: Implementación de K-Means para Análisis de Calidad de Vino
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

from src.data import PaqueteDatos, normalizar_paquete, cargar_calidad_vino, EscaladorZScore
from src.evaluation import evaluar_modelos
from src.kmeans_loop import KMeansLoop
from src.kmeans_numpy import KMeansNumpy
from src.kmeans_sklearn import KMeansSklearn
from src.visualization import graficar_pca_2d, graficar_radar_centroides, graficar_dispersion_2d, graficar_dispersion_3d


st.set_page_config(
    page_title="SIA - Analisis K-Means",
    layout="wide",
    page_icon=None
)

# --- Encabezado & Barra Lateral ---
st.title("Sistema de Análisis de Calidad Vitivinícola")
st.markdown("**Departamento de Inteligencia Artificial - Ultralistic**")
st.markdown("Jesús Oliveto")
st.markdown("---")

st.sidebar.title("Configuración")
st.sidebar.markdown("**Panel de Control**")

@st.cache_data
def cargar_datos(nombre_archivo: str = "winequalityclean.arff") -> Tuple[PaqueteDatos, PaqueteDatos]:
    """
    Rutina de carga de datos con caché en memoria.
    
    ¿Qué hace?:
    Lee el archivo ARFF especificado, separa las características de los labels
    y devuelve dos versiones del dataset: una cruda (original) y otra estandarizada (Z-Score).
    
    ¿Cómo lo hace?:
    Aprovecha el decorador `@st.cache_data` de Streamlit para evitar recargar el archivo 
    del disco duro en cada interacción (recálculo) de la interfaz gráfica. Llama internamente
    a `cargar_calidad_vino` y `normalizar_paquete` del módulo `data.py`.
    
    Finalidad:
    Minimizar la latencia I/O de la aplicación web y asegurar que todos los algoritmos 
    puedan comparar la diferencia entre procesar datos crudos vs datos escalados estadísticamente.

    Args:
        nombre_archivo: Nombre del archivo ARFF en el directorio 'datasets'.
        
    Returns:
        Tupla de (PaqueteDatos Crudo, PaqueteDatos Normalizado).
    """
    ruta_datos = Path(__file__).resolve().parent / "datasets" / nombre_archivo
    paquete = cargar_calidad_vino(ruta_datos)
    paquete_norm, escalador = normalizar_paquete(paquete)
    paquete_norm.escalador = escalador  # type: ignore[attr-defined]
    return paquete, paquete_norm


def construir_implementaciones() -> Dict[str, str]:
    """
    Provisión de las variantes del algoritmo.
    
    ¿Qué hace?: 
    Devuelve un diccionario que asocia claves internas con etiquetas de interfaz de usuario.
    
    ¿Cómo lo hace?: 
    Simplemente mapea strings ("loop", "numpy", "sklearn") a sus nombres legibles
    para popular los selectores múltiples (multiselect) del sidebar.
    
    Finalidad:
    Cargar de manera programática en la UI las diferencias que el proyecto 
    pretende mostrar, permitiendo al usuario decidir qué motores correr.
    """
    return {
        "loop": "K-Means (Bucles Python)",
        "numpy": "K-Means (NumPy Vectorizado)",
        "sklearn": "K-Means (Estándar Scikit-Learn)"
    }


def fabrica_constructores(nombre: str):
    """
    Patrón de diseño Factory para el despacho de instanciación K-Means.
    
    ¿Qué hace?:
    Retorna una función generadora (closure) que construye dinámicamente el modelo
    apropiado basado en el nombre string solicitado por la UI.
    
    ¿Cómo lo hace?:
    Usa estructuras `if/elif` para devolver una función lambda que, al momento de ser
    invocada con `(k, semilla)`, enlazará todos los parámetros (ej. `num_inicios`) correctos para 
    esa implementación subyacente (`KMeansLoop`, `KMeansNumpy`, etc.).
    
    Finalidad:
    Desacoplar la UI de la creación directa de objetos. Permite al motor de evaluación
    instanciar algoritmos ciegamente usando una misma firma (interface polimórfica), 
    facilitando el "benchmarking".
    
    Args:
        nombre: Clave de la implementación ('loop', 'numpy', 'sklearn').
        
    Returns:
        Un callable que al ejecutarse devuelve el modelo listo para ser ajustado.
    """
    if nombre == "loop":
        return lambda k, semilla, detallado=False: KMeansLoop(num_clusters=k, num_inicios=3, estado_aleatorio=semilla, detallado=detallado)
    if nombre == "numpy":
        return lambda k, semilla, detallado=False: KMeansNumpy(num_clusters=k, num_inicios=3, estado_aleatorio=semilla, detallado=detallado)
    if nombre == "sklearn":
        return lambda k, semilla, detallado=False: KMeansSklearn(num_clusters=k, num_inicios=10, estado_aleatorio=semilla)
    raise ValueError(f"Implementación no soportada: {nombre}")


def ejecutar_evaluacion(paquete_norm: PaqueteDatos, ks: List[int], claves_impl: List[str], num_corridas: int, con_silhouette: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Invocación masiva del núcleo evaluador.
    
    ¿Qué hace?:
    Ejecuta el bloque de entrenamiento y métricas (`evaluar_modelos`) sobre todas 
    las 'K' (clusters) y motores seleccionados, tabulando los resultados sin procesar
    y promediándolos para la presentación.
    
    ¿Cómo lo hace?:
    Adapta las funciones `factory` locales, lanza la validación cruzada y consolida 
    las estructuras generadas (`ResultadoEjecucion`) en un DataFrame de Pandas para aplicar 
    cálculos agregados (GroupBy `mean`) sobre el Tiempo, Inercia, ARI, etc.
    
    Finalidad:
    Servir como puente de integración de Pandas a Streamlit. Prepara la "tabla gorda" de
    todas las corridas de algoritmos en bruto (`df_corridas`) y la "tabla limpia" 
    estadísticamente promediada (`df_resumen`) que consume la solapa Comparativa.
    """
    constructores_adaptados = {
        k: lambda n_k, s: fabrica_constructores(k)(n_k, s, detallado=False) 
        for k in claves_impl
    }
    
    resultados = evaluar_modelos(
        datos=paquete_norm.datos,
        valores_k=ks,
        constructores=constructores_adaptados,
        num_corridas=num_corridas,
        estado_aleatorio=42,
        calcular_silhouette=con_silhouette,
        etiquetas_reales=paquete_norm.etiquetas_reales
    )
    
    filas = []
    for r in resultados:
        filas.append({
            "Implementación": r.implementacion,
            "k": r.k,
            "Corrida": r.corrida,
            "Inercia": r.inercia,
            "Tiempo (s)": r.tiempo_entrenamiento,
            "Iteraciones": r.num_iteraciones,
            "Silhouette": r.silhouette,
            "ARI": r.ari,
            "NMI": r.nmi
        })
        
    df = pd.DataFrame(filas)
    
    diccionario_agregacion = {"Inercia": "mean", "Tiempo (s)": "mean", "Iteraciones": "mean", "Silhouette": "mean"}
    if paquete_norm.etiquetas_reales is not None:
        diccionario_agregacion["ARI"] = "mean"
        diccionario_agregacion["NMI"] = "mean"
        
    resumen = (
        df.groupby(["Implementación", "k"])
        .agg(diccionario_agregacion)
        .reset_index()
    )
    return df, resumen


def predecir_muestra(impl: str, k: int, muestra: np.ndarray, paquete_norm: PaqueteDatos, estado_aleatorio: int):
    """
    Inferencia end-to-end de una nueva muestra definida interactivamente en la interfaz.
    
    ¿Qué hace?:
    Entrena por completo el modelo K-Means deseado sobre el dataset histórico base, 
    evalúa la nueva muestra y calcula a qué distancia terminó clasificándose respecto 
    de todos los centroides formados.
    
    ¿Cómo lo hace?:
    Llama a la `fabrica_constructores`, encadena un `ajustar()` completo de los datos, 
    y luego invoca a `predecir()`. Por último, usa el álgebra de norma Euclidiana de 
    Numpy (`np.linalg.norm`) contra la memoria de los centroides de la clase resultante.
    
    Finalidad:
    Demostrar visualmente y de forma aislada el comportamiento "en vivo" (en "producción")
    del algoritmo, calculando distancias sin trampas de caché tal como indican 
    los requerimientos explicativos de la simulación.
    """
    # 1. Instanciar y Entrenar (Desde cero)
    modelo = fabrica_constructores(impl)(k, estado_aleatorio, detallado=False)
    modelo.ajustar(paquete_norm.datos)
    
    # 2. Predecir
    etiquetas_ = modelo.predecir(muestra)
    distancias = np.linalg.norm(modelo.centroides_ - muestra, axis=1)
    
    return int(etiquetas_[0]), distancias, modelo


# --- Entradas de Barra Lateral ---
st.sidebar.subheader("Selección de Datos")
mapa_datasets = {
    "winequalityclean.arff": "Dataset Limpio (Recomendado)",
    "winequality.arff": "Dataset Original (Con Outliers)"
}
archivo_seleccionado = st.sidebar.selectbox(
    "Archivo de Origen",
    options=list(mapa_datasets.keys()),
    format_func=lambda x: mapa_datasets[x],
    index=0
)

# Cargar Datos
try:
    with st.spinner("Cargando dataset..."):
        paquete_crudo, paquete_norm = cargar_datos(archivo_seleccionado)
    st.sidebar.success(f"Cargado: {len(paquete_crudo.datos)} registros, {len(paquete_crudo.nombres_caracteristicas)} atributos")
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()


st.sidebar.subheader("Parámetros Globales")
implementaciones = construir_implementaciones()
impls_seleccionadas = st.sidebar.multiselect(
    "Implementaciones a Comparar",
    options=list(implementaciones.keys()),
    format_func=lambda k: implementaciones[k],
    default=list(implementaciones.keys()),
)

rango_k = st.sidebar.slider("Rango de Clusters (k)", 2, 12, (3, 8))
num_corridas_ui = st.sidebar.slider("Corridas por Configuración", 1, 5, 2, help="Promediar resultados para mayor robustez.")
con_silhouette_ui = st.sidebar.checkbox("Calcular Silhouette Score", value=True)

st.sidebar.subheader("Selección de Atributos")
atributos_seleccionados = st.sidebar.multiselect(
    "Atributos a utilizar en el algoritmo",
    options=paquete_crudo.nombres_caracteristicas,
    default=paquete_crudo.nombres_caracteristicas,
    help="Seleccione al menos 2 atributos. Con 2 o 3 se graficará directamente; con más se usará PCA."
)

if len(atributos_seleccionados) < 2:
    st.sidebar.error("⚠️ Seleccione al menos 2 atributos.")

# Computar datos filtrados según atributos
indices_caracteristicas = [paquete_crudo.nombres_caracteristicas.index(f) for f in atributos_seleccionados]
datos_x_seleccionados = paquete_norm.datos[:, indices_caracteristicas] if indices_caracteristicas else paquete_norm.datos
nombres_caracteristicas_seleccionadas = atributos_seleccionados if atributos_seleccionados else paquete_crudo.nombres_caracteristicas


# --- Solapas (Tabs) ---
tab_explorar, tab_codo, tab_comparar, tab_predecir, tab_depurar = st.tabs([
    "0. Explorador de Datos",
    "1. Método del Codo",
    "2. Comparativa",
    "3. Predicción",
    "4. Análisis Detallado"
])

# --- 0. Explorador de Datos ---
with tab_explorar:
    st.header("Explorador de Datos")
    st.markdown("Inspeccione los datos antes de ejecutar el algoritmo. Cambie entre la vista cruda y normalizada para verificar el pre-procesamiento.")

    modo_vista = st.radio("Vista", ["Datos Crudos", "Datos Normalizados"], horizontal=True)

    if modo_vista == "Datos Crudos":
        df_vista = pd.DataFrame(paquete_crudo.datos, columns=paquete_crudo.nombres_caracteristicas)
        st.caption(f"Mostrando datos originales — {len(df_vista)} registros, {len(paquete_crudo.nombres_caracteristicas)} atributos")
    else:
        df_vista = pd.DataFrame(paquete_norm.datos, columns=paquete_crudo.nombres_caracteristicas)
        st.caption(f"Mostrando datos normalizados (Z-Score) — {len(df_vista)} registros, {len(paquete_crudo.nombres_caracteristicas)} atributos")

    # Agregar columna de calidad (clase) si está disponible
    if paquete_crudo.etiquetas_reales is not None:
        df_vista.insert(0, "Calidad", paquete_crudo.etiquetas_reales)

    st.dataframe(df_vista, use_container_width=True, height=280)

    col_stats1, col_stats2 = st.columns([2, 1])

    with col_stats1:
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(
            df_vista.describe().T.style.format("{:.4f}"),
            use_container_width=True
        )

    with col_stats2:
        st.subheader("Atributo Individual")
        atributo_elegido = st.selectbox("Seleccionar atributo", options=paquete_crudo.nombres_caracteristicas)
        if atributo_elegido:
            col_datos = df_vista[atributo_elegido]
            st.metric("Nombre", atributo_elegido)
            st.metric("Mínimo", f"{col_datos.min():.5f}")
            st.metric("Máximo", f"{col_datos.max():.5f}")
            st.metric("Media", f"{col_datos.mean():.5f}")
            st.metric("Desv. Estándar", f"{col_datos.std():.5f}")

# --- 1. Método del Codo ---
with tab_codo:
    st.header("Determinación de K óptimo")
    st.info("Utilice el **Método del Codo** para identificar el número óptimo de clusters. Busque el punto donde la ganancia de inercia disminuye drásticamente.")
    
    col_e1, col_e2 = st.columns([1, 3])
    with col_e1:
        k_codo_maximo = st.number_input("K Máximo", min_value=5, max_value=50, value=15)
        gatillo_codo = st.button("Generar Gráfico", type="primary", key="btn_codo")
        
    with col_e2:
        if gatillo_codo:
            if len(atributos_seleccionados) < 2:
                st.warning("Seleccione al menos 2 atributos en el sidebar para continuar.")
            else:
              with st.spinner("Calculando curva de inercia..."):
                datos_codo = []
                fabrica = fabrica_constructores("numpy")

                barra_progreso = st.progress(0)
                for i, valor_k in enumerate(range(1, k_codo_maximo + 1)):
                    modelo_codo = fabrica(valor_k, 42, detallado=False)
                    modelo_codo.ajustar(datos_x_seleccionados)
                    datos_codo.append({"k": valor_k, "Inercia": modelo_codo.inercia_})
                    barra_progreso.progress((i + 1) / k_codo_maximo)
                
                df_codo = pd.DataFrame(datos_codo)
                
                fig_codo = px.line(
                    df_codo, 
                    x="k", 
                    y="Inercia", 
                    title="Análisis del Codo (Inercia vs k)",
                    markers=True,
                    labels={"k": "Número de Clusters (k)", "Inercia": "Suma de Errores al Cuadrado (SSE)"}
                )
                fig_codo.update_layout(xaxis=dict(dtick=1))
                st.plotly_chart(fig_codo, use_container_width=True)

# --- 2. Comparativa ---
with tab_comparar:
    st.header("Comparativa de Rendimiento y Calidad")
    st.markdown("Ejecute múltiples implementaciones para validar que los resultados sean consistentes y comparar tiempos de ejecución.")
    
    if st.button("Iniciar Comparativa", key="btn_comparar"):
        if not impls_seleccionadas:
            st.warning("Por favor seleccione al menos una implementación en la barra lateral.")
        else:
            if len(atributos_seleccionados) < 2:
                st.warning("Seleccione al menos 2 atributos en el sidebar para continuar.")
            else:
              valores_ks = list(range(rango_k[0], rango_k[1] + 1))
              import dataclasses
              paquete_norm_filtrado = dataclasses.replace(paquete_norm, datos=datos_x_seleccionados, nombres_caracteristicas=nombres_caracteristicas_seleccionadas)
              with st.spinner("Ejecutando benchmarks..."):
                df_corridas, df_resumen = ejecutar_evaluacion(paquete_norm_filtrado, valores_ks, impls_seleccionadas, num_corridas_ui, con_silhouette_ui)
            
            st.success("Evaluación completada con éxito.")
            
            st.subheader("Resumen General")
            st.markdown(
                """
                **Interpretación de Métricas:**
                - **Inercia (SSE):** Cohesión interna (Menor es mejor).
                - **Silhouette:** Definición de clusters (-1 a 1, Mayor es mejor).
                - **ARI (Adjusted Rand Index):** Coincidencia con la calidad real del vino (0 a 1, Mayor es mejor). Mide la utilidad real.
                - **NMI (Normalized Mutual Information):** Información mutua normalizada entre clusters y calidad real (0 a 1, Mayor es mejor). Cuantifica cuánta información sobre la calidad retienen los clusters.
                - **Tiempo:** Eficiencia computacional.
                """
            )

            diccionario_fmt = {
                "Inercia": "{:.2f}", 
                "Tiempo (s)": "{:.6f}", 
                "Silhouette": "{:.3f}",
                "Iteraciones": "{:.1f}"
            }
            if "ARI" in df_resumen.columns:
                diccionario_fmt["ARI"] = "{:.3f}"
            if "NMI" in df_resumen.columns:
                diccionario_fmt["NMI"] = "{:.3f}"

            st.dataframe(
                df_resumen.style.format(diccionario_fmt)
                .background_gradient(subset=["Tiempo (s)"], cmap="RdYlGn_r")
                .background_gradient(subset=["Inercia"], cmap="Blues_r")
                .highlight_max(subset=["Silhouette"], color="#d4edda"),
                use_container_width=True
            )
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Tiempo de Ejecución vs K")
                st.line_chart(df_resumen.pivot(index="k", columns="Implementación", values="Tiempo (s)"))
                
                if "numpy" in impls_seleccionadas and "loop" in impls_seleccionadas:
                    tiempos_np = df_resumen[df_resumen["Implementación"] == "numpy"].set_index("k")["Tiempo (s)"]
                    tiempos_loop = df_resumen[df_resumen["Implementación"] == "loop"].set_index("k")["Tiempo (s)"]
                    aceleracion = tiempos_loop / tiempos_np
                    
                    st.markdown("#### Speedup (Bucle vs Numpy)")
                    st.bar_chart(aceleracion)
                    st.caption("Factor de aceleración: Cuántas veces más rápido es NumPy vs Bucles Python.")

            with c2:
                if "ARI" in df_resumen.columns:
                    st.markdown("#### Validación Externa (ARI) vs K")
                    st.line_chart(df_resumen.pivot(index="k", columns="Implementación", values="ARI"))
                    st.caption("¿Los clusters coinciden con la calidad del vino? (ARI > 0 indica correlación)")
                else:
                    st.markdown("#### Inercia vs K")
                    st.line_chart(df_resumen.pivot(index="k", columns="Implementación", values="Inercia"))

# --- 3. Predicción ---
with tab_predecir:
    st.header("Simulación de Predicción")
    st.markdown("Simule la llegada de una nueva muestra de vino. El sistema **re-entrenará el modelo** con todo el dataset y clasificará la muestra.")
    
    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        st.subheader("Configuración del Modelo")
        p_impl = st.selectbox("Algoritmo", list(implementaciones.keys()), format_func=lambda k: implementaciones[k])
        p_k = st.slider("Número de Clusters (k)", 2, 12, 4, key="pred_k")
        p_semilla = st.number_input("Semilla Aleatoria", value=42, step=1)
        
        btn_predecir = st.button("Clasificar Muestra", type="primary")
        
    with col_p2:
        st.subheader("Atributos de la Muestra")
        st.caption("Modifique los valores para definir el nuevo vino.")
        
        valores_entrada = []
        columnas_ui = st.columns(3)
        for i, nombre_attr in enumerate(paquete_crudo.nombres_caracteristicas):
            valor_defecto = float(np.mean(paquete_crudo.datos[:, i]))
            with columnas_ui[i % 3]:
                valor_ingresado = st.number_input(nombre_attr, value=valor_defecto, format="%.4f")
                valores_entrada.append(valor_ingresado)
                
    if btn_predecir:
        if len(atributos_seleccionados) < 2:
            st.warning("Seleccione al menos 2 atributos en el sidebar para continuar.")
        else:
          muestra_array = np.array(valores_entrada, dtype=np.float64).reshape(1, -1)

          escalador = getattr(paquete_norm, "escalador")
          muestra_normalizada = escalador.transformar(muestra_array)  # type: ignore[assignment]
          muestra_normalizada = muestra_normalizada[:, indices_caracteristicas]
        
          import dataclasses
          paquete_pred = dataclasses.replace(paquete_norm, datos=datos_x_seleccionados, nombres_caracteristicas=nombres_caracteristicas_seleccionadas)

          with st.status("Procesando...", expanded=True) as status:
              st.write("1. Normalizando datos de entrada...")
              time.sleep(0.3)

              st.write(f"2. Entrenando modelo {implementaciones[p_impl]} con k={p_k} desde cero...")
              t_inicio = time.perf_counter()
              etiqueta_asignada, distancias_centro, _ = predecir_muestra(p_impl, p_k, muestra_normalizada, paquete_pred, int(p_semilla))
              t_fin = time.perf_counter()
              st.write(f"Ref: Modelo entrenado en {t_fin - t_inicio:.4f}s")

              st.write("3. Calculando distancias a centroides finales...")
              status.update(label="Clasificación Completada", state="complete", expanded=False)

          # Calcular calidad promedio por cluster
          modelo_pred = fabrica_constructores(p_impl)(p_k, int(p_semilla), detallado=False)
          modelo_pred.ajustar(paquete_pred.datos)
          etiquetas_entrenamiento = modelo_pred.etiquetas_

          calidad_por_cluster = {}
          if paquete_crudo.etiquetas_reales is not None:
              for cid in range(p_k):
                  mascara = etiquetas_entrenamiento == cid
                  if mascara.any():
                      calidad_por_cluster[cid] = float(paquete_crudo.etiquetas_reales[mascara].mean())
                  else:
                      calidad_por_cluster[cid] = None

          col_res1, col_res2, col_res3 = st.columns([1, 1, 1])
          with col_res1:
              st.metric("Cluster Asignado", f"Cluster {etiqueta_asignada}")
              st.info(f"El vino ha sido clasificado en el Grupo {etiqueta_asignada}.")

          with col_res2:
              st.markdown(f"**Distancia mínima:** {distancias_centro[etiqueta_asignada]:.4f}")

          with col_res3:
              if etiqueta_asignada in calidad_por_cluster and calidad_por_cluster[etiqueta_asignada] is not None:
                  calidad_pred = calidad_por_cluster[etiqueta_asignada]
                  st.metric("Calidad Predicha", f"{calidad_pred:.2f} / 10")
                  st.caption("Promedio de calidad de los vinos en este cluster.")
              else:
                  st.metric("Calidad Predicha", "N/D")

          df_distancias = pd.DataFrame({
              "Cluster ID": range(p_k),
              "Distancia Euclidiana": distancias_centro,
              "Calidad Promedio": [f"{calidad_por_cluster.get(i, 'N/D'):.2f}" if calidad_por_cluster.get(i) is not None else "N/D" for i in range(p_k)],
              "Muestras": [int((etiquetas_entrenamiento == i).sum()) for i in range(p_k)],
              "Estado": ["✅ ASIGNADO" if i == etiqueta_asignada else "-" for i in range(p_k)]
          })
          st.table(df_distancias.style.highlight_min(subset=["Distancia Euclidiana"], color="#d4edda", axis=0))

# --- 4. Análisis Profundo ---
with tab_depurar:
    st.header("Interpretación y Depuración")
    
    col_d1, col_d2 = st.columns([1, 2])
    
    with col_d1:
        st.markdown("##### Configuración")
        d_impl = st.selectbox("Algoritmo", ["loop", "numpy", "sklearn"], key="d_impl", format_func=lambda x: implementaciones[x])
        d_k = st.slider("k", 2, 10, 4, key="d_k")
        d_semilla = st.number_input("Semilla", value=42, key="d_semilla")
        
        st.markdown("##### Modo")
        modo_depuracion = st.radio("Tipo de Análisis", ["Perfilado (Heatmap)", "Depuración (Logs)"])
        btn_ejecutar_analisis = st.button("Ejecutar Análisis")
        
    with col_d2:
        if btn_ejecutar_analisis:
            if len(atributos_seleccionados) < 2:
                st.warning("Seleccione al menos 2 atributos en el sidebar para continuar.")
            elif modo_depuracion == "Depuración (Logs)":
                st.markdown("#### Logs de Ejecución Paso a Paso")
                captura_log = io.StringIO()

                with st.spinner("Ejecutando con logging activado..."):
                    with contextlib.redirect_stdout(captura_log):
                        modelo_dep = fabrica_constructores(d_impl)(d_k, int(d_semilla), detallado=True)
                        if hasattr(modelo_dep, 'num_inicios'):
                            modelo_dep.num_inicios = 1

                        modelo_dep.ajustar(datos_x_seleccionados)

                st.text_area("Traza del Algoritmo", captura_log.getvalue(), height=400)

            else: 
                st.markdown("#### Perfilado de Clusters (Heatmap)")
                with st.spinner("Generando mapa de calor..."):
                    modelo_perf = fabrica_constructores(d_impl)(d_k, int(d_semilla), detallado=False)
                    modelo_perf.ajustar(datos_x_seleccionados)

                    col_nombres_selec = nombres_caracteristicas_seleccionadas

                    # Calcular calidad promedio por cluster
                    calidades_cluster = []
                    if paquete_crudo.etiquetas_reales is not None:
                        for cid in range(d_k):
                            mascara = modelo_perf.etiquetas_ == cid
                            if mascara.any():
                                calidades_cluster.append(float(paquete_crudo.etiquetas_reales[mascara].mean()))
                            else:
                                calidades_cluster.append(0.0)

                    df_centroides = pd.DataFrame(modelo_perf.centroides_, columns=col_nombres_selec)
                    df_centroides.index.name = "Cluster"

                    # Agregar calidad promedio al heatmap
                    if calidades_cluster:
                        escalador_calidad = paquete_crudo.etiquetas_reales.max() if paquete_crudo.etiquetas_reales is not None else 10
                        df_centroides["Calidad Prom."] = [c / escalador_calidad for c in calidades_cluster]
                        col_nombres_heatmap = list(col_nombres_selec) + ["Calidad Prom."]
                    else:
                        col_nombres_heatmap = list(col_nombres_selec)

                    fig_calor = px.imshow(
                        df_centroides,
                        labels=dict(x="Característica", y="Cluster", color="Z-Score"),
                        x=col_nombres_heatmap,
                        y=[f"Cluster {i}" for i in range(d_k)],
                        color_continuous_scale="RdBu_r",
                        aspect="auto",
                        title="Mapa de Calor de Centroides (Estandarizado Z-Score)"
                    )
                    st.plotly_chart(fig_calor, use_container_width=True)
                    st.caption("Colores Rojos indican valores por encima del promedio. Azules indican por debajo.")

                # --- Tabla de Calidad por Cluster ---
                if paquete_crudo.etiquetas_reales is not None:
                    st.markdown("#### Calidad del Vino por Cluster")
                    filas_calidad = []
                    for cid in range(d_k):
                        mascara = modelo_perf.etiquetas_ == cid
                        clases_en_cluster = paquete_crudo.etiquetas_reales[mascara]
                        if len(clases_en_cluster) > 0:
                            filas_calidad.append({
                                "Cluster": f"Cluster {cid}",
                                "Muestras": int(mascara.sum()),
                                "Calidad Promedio": f"{clases_en_cluster.mean():.2f}",
                                "Calidad Mínima": int(clases_en_cluster.min()),
                                "Calidad Máxima": int(clases_en_cluster.max()),
                                "Desv. Estándar": f"{clases_en_cluster.std():.2f}",
                            })
                    df_calidad = pd.DataFrame(filas_calidad)
                    st.dataframe(df_calidad, use_container_width=True, hide_index=True)
                    st.caption("La calidad promedio indica qué nivel de vino tiende a agruparse en cada cluster (escala 1-10).")

                st.markdown("#### Visualización Espacial")
                num_sel = len(nombres_caracteristicas_seleccionadas)
                if num_sel == 2:
                    etiquetas_tabs = ["Scatter 2D Directo", "Radar Chart"]
                elif num_sel == 3:
                    etiquetas_tabs = ["Scatter 3D Directo", "Radar Chart"]
                else:
                    etiquetas_tabs = ["PCA 2D", "Radar Chart"]

                t_dispersion, t_radar = st.tabs(etiquetas_tabs)
                with t_dispersion:
                    if num_sel == 2:
                        st.plotly_chart(graficar_dispersion_2d(datos_x_seleccionados, modelo_perf.etiquetas_, nombres_caracteristicas_seleccionadas), use_container_width=True)
                        st.caption("Gráfico directo: los ejes representan los 2 atributos seleccionados (sin reducción de dimensionalidad).")
                    elif num_sel == 3:
                        st.plotly_chart(graficar_dispersion_3d(datos_x_seleccionados, modelo_perf.etiquetas_, nombres_caracteristicas_seleccionadas), use_container_width=True)
                        st.caption("Gráfico directo 3D interactivo: los ejes representan los 3 atributos seleccionados.")
                    else:
                        st.plotly_chart(graficar_pca_2d(datos_x_seleccionados, modelo_perf.etiquetas_), use_container_width=True)
                        st.caption(f"Proyección PCA 2D aplicada sobre los {num_sel} atributos seleccionados.")
                with t_radar:
                    st.plotly_chart(graficar_radar_centroides(modelo_perf.centroides_, nombres_caracteristicas_seleccionadas), use_container_width=True)
