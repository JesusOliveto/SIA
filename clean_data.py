"""
Script de preprocesamiento y limpieza de datos (ETL: Extract, Transform, Load).

Finalidad:
Preparar un conjunto de datos robusto depurando valores atípicos (outliers) antes 
del entrenamiento de K-Means. K-Means es extremadamente sensible a outliers
porque minimiza distancias cuadráticas, lo que hace que un valor extremo arrastre 
desproporcionadamente a los centroides y arruine los clústeres.
"""
import logging
import sys
from pathlib import Path

import pandas as pd
from scipy.io import arff

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cargar_arff(ruta: Path) -> pd.DataFrame:
    """
    Carga de datos crudos.
    
    ¿Qué hace?: 
    Lee un archivo con formato ARFF y lo convierte en una estructura tabular.
    
    ¿Cómo lo hace?: 
    Utiliza el parser `arff.loadarff` de SciPy y envuelve el resultado en un 
    DataFrame de Pandas. Además, decodifica cadenas binarias a UTF-8 si es necesario.
    
    Finalidad: 
    Ingesta inicial de los datos de la base de datos estática al pipeline en memoria.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {ruta}")
    
    logger.info(f"Cargando dataset original desde {ruta}...")
    crudo, _ = arff.loadarff(ruta)
    df = pd.DataFrame(crudo)
    
    # Decodificar columnas tipo byte a string (común en arff)
    for col in df.select_dtypes([object]).columns:
        if isinstance(df[col].iloc[0], bytes):
            df[col] = df[col].str.decode('utf-8')
            
    return df

def limpiar_atipicos(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Detección y eliminación de valores atípicos (Outliers).
    
    ¿Qué hace?:
    Descarta muestras que presenten valores estadísticamente extremos en cualquiera 
    de sus atributos numéricos.
    
    ¿Cómo lo hace?:
    Aplica el método del Rango Intercuartílico (IQR). Calcula el cuartil 1 (Q1, 25%) y 
    el cuartil 3 (Q3, 75%). Define los límites como `Q1 - factor * IQR` y 
    `Q3 + factor * IQR`. Usa Pandas para filtrar y retener sólo las filas dentro de ese rango.
    
    Finalidad:
    Como los centroides en K-Means son el promedio aritmético de sus puntos, suprimir 
    estos "tirones" gravitacionales indeseados resulta en esferas de cluster más compactas 
    y realistas, aumentando dramáticamente la silueta y significancia del agrupamiento.

    Args:
        df: DataFrame de entrada.
        factor: Multiplicador del IQR (por defecto 1.5, estándar estadístico).
        
    Returns:
        DataFrame limpio.
    """
    logger.info("Detectando outliers usando método IQR...")
    
    # Excluir la columna categórica de clase al detectar numéricamente
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    if 'class' in columnas_numericas:
        columnas_numericas = columnas_numericas.drop('class')

    filas_iniciales = len(df)
    mascara_limpia = pd.Series(True, index=df.index)

    for col in columnas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        limite_inferior = Q1 - factor * IQR
        limite_superior = Q3 + factor * IQR
        
        mascara_columna = (df[col] >= limite_inferior) & (df[col] <= limite_superior)
        mascara_limpia &= mascara_columna

    df_limpio = df[mascara_limpia].copy()
    
    filas_removidas = filas_iniciales - len(df_limpio)
    logger.info(f"Removidas {filas_removidas} filas conteniendo al menos un outlier. ({filas_removidas/filas_iniciales*100:.1f}%)")
    logger.info(f"Forma original: {df.shape}, Forma limpia: {df_limpio.shape}")
    
    return df_limpio

def guardar_arff(df: pd.DataFrame, ruta_entrada: Path, ruta_salida: Path) -> None:
    """
    Serialización de datos limpios.
    
    ¿Qué hace?:
    Vuelca el DataFrame procesado de nuevo al disco físico en el formato ARFF original.
    
    ¿Cómo lo hace?:
    Es una escritura híbrida. Abre el archivo original (ruta_entrada) únicamente para "robar"
    la cabecera (los tags @RELATION y @ATTRIBUTE). Luego itera sobre los registros limpios 
    de Pandas y los añade al nuevo archivo como líneas de valores separados por comas.
    
    Finalidad:
    Generar el artefacto final (`winequalityclean.arff`) que la aplicación web consumirá 
    durante el arranque, garantizando la persistencia de la limpieza.

    Args:
        df: DataFrame a guardar.
        ruta_entrada: Ruta del archivo ARFF original (para extraer Metadata).
        ruta_salida: Ruta de escritura del nuevo dataset limpio.
    """
    logger.info(f"Guardando datos limpios en {ruta_salida}...")
    
    # 1. Extraer la cabecera original (Relación y Atributos)
    cabecera_arff = []
    with open(ruta_entrada, 'r', encoding='utf-8') as f:
        for linea in f:
            if linea.upper().startswith('@DATA'):
                cabecera_arff.append(linea)
                break
            cabecera_arff.append(linea)

    # 2. Escribir cabecera y datos
    with open(ruta_salida, 'w', encoding='utf-8') as f:
        f.writelines(cabecera_arff)
        
        for i, fila in df.iterrows():
            valores = []
            for col in df.columns:
                val = fila[col]
                if isinstance(val, (int, float)):
                    # Formatear números
                    valores.append(str(val))
                else:
                    # Strings/Clases
                    valores.append(str(val))
            f.write(','.join(valores) + '\n')
            
    logger.info("Guardado completado exitosamente.")

def main():
    """
    Orquestador principal del proceso de limpieza.
    
    ¿Qué hace?:
    Une secuencialmente la Ingesta -> Limpieza -> Guardado en disco.
    Define las constantes de directorio y lanza el trabajo.
    """
    directorio_base = Path("datasets")
    archivo_entrada = directorio_base / "winequality.arff"
    archivo_salida = directorio_base / "winequalityclean.arff" 
    
    if not archivo_entrada.exists():
        logger.error(f"Archivo de entrada no encontrado: {archivo_entrada}")
        return
        
    try:
        df_crudo = cargar_arff(archivo_entrada)
        df_limpio = limpiar_atipicos(df_crudo)
        guardar_arff(df_limpio, archivo_entrada, archivo_salida)
        logger.info("Pipeline de limpieza finalizado.")
    except Exception as e:
        logger.error(f"El pipeline falló: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
