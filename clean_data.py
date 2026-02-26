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
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.io import arff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_arff(path: Path) -> pd.DataFrame:
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
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info(f"Loading data from {path}...")
    try:
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        
        # Decode bytes columns to strings if necessary
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
                except Exception as e:
                    logger.warning(f"Could not decode column {col}: {e}")
        
        return df
    except Exception as e:
        logger.error(f"Failed to load ARFF file: {e}")
        raise

def clean_outliers(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
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
    logger.info("Detecting outliers using IQR method...")
    
    # Select only numeric columns for outlier detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Exclude 'class' column if it exists and is numeric (it shouldn't be touched)
    if 'class' in numeric_cols:
        numeric_cols = numeric_cols.drop('class')
        
    logger.info(f"Analyzing features: {list(numeric_cols)}")
    
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # Keep rows where ALL features are within bounds
    # Alternatively: Remove row if ANY feature is an outlier
    condition = ~((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound)).any(axis=1)
    
    df_clean = df[condition].copy()
    
    removed_count = len(df) - len(df_clean)
    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Cleaned shape: {df_clean.shape}")
    logger.info(f"Removed {removed_count} rows ({removed_count/len(df):.2%}).")
    
    return df_clean

def save_arff(df: pd.DataFrame, input_path: Path, output_path: Path) -> None:
    """
    Serialización de datos limpios.
    
    ¿Qué hace?:
    Vuelca el DataFrame procesado de nuevo al disco físico en el formato ARFF original.
    
    ¿Cómo lo hace?:
    Es una escritura híbrida. Abre el archivo original (input_path) únicamente para "robar"
    la cabecera (los tags @RELATION y @ATTRIBUTE). Luego itera sobre los registros limpios 
    de Pandas y los añade al nuevo archivo como líneas de valores separados por comas.
    
    Finalidad:
    Generar el artefacto final (`whinequalityclean.arff`) que la aplicación web consumirá 
    durante el arranque, garantizando la persistencia de la limpieza.

    Args:
        df: DataFrame a guardar.
        input_path: Ruta del archivo ARFF original (para extraer Metadata).
        output_path: Ruta de escritura del nuevo dataset limpio.
    """
    logger.info(f"Saving cleaned data to {output_path}...")
    
    try:
        with open(input_path, 'r') as f_in:
            content = f_in.readlines()
        
        # Locate @DATA tag
        data_start_idx = 0
        for i, line in enumerate(content):
            if line.strip().upper().startswith("@DATA"):
                data_start_idx = i + 1
                break
        
        header = content[:data_start_idx]
        
        with open(output_path, 'w') as f_out:
            f_out.writelines(header)
            
            # Write data rows
            # We iterate to ensure correct formatting (strings vs numbers)
            for _, row in df.iterrows():
                line_parts = []
                for col in df.columns:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        line_parts.append(str(val))
                    else:
                        line_parts.append(str(val))
                f_out.write(",".join(line_parts) + "\n")
                
        logger.info("Save successful.")
        
    except Exception as e:
        logger.error(f"Failed to save ARFF file: {e}")
        raise

def main():
    """
    Orquestador principal del proceso de limpieza.
    
    ¿Qué hace?:
    Une secuencialmente la Ingesta -> Limpieza -> Guardado en disco.
    Define las constantes de directorio y lanza el trabajo.
    """
    base_dir = Path("datasets")
    input_file = base_dir / "winequality.arff"
    output_file = base_dir / "winequalityclean.arff" 
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    try:
        df = load_arff(input_file)
        df_clean = clean_outliers(df)
        save_arff(df_clean, input_file, output_file)
        logger.info("Data cleaning process completed.")
    except Exception:
        logger.error("Process failed.")

if __name__ == "__main__":
    main()
