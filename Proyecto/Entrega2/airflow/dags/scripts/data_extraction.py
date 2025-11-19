"""
Módulo de Extracción y Validación de Datos - SodAI Drinks
=========================================================

Este módulo se encarga de la extracción y validación de los datos de entrada
del sistema de recomendación de SodAI Drinks.

Datos manejados:
- transacciones.parquet: Transacciones históricas de los clientes
- clientes.parquet: Información demográfica y geográfica de clientes  
- productos.parquet: Catálogo de productos con categorías y marcas

Autor: SodAI Drinks MLOps Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
import os
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataExtractor:
    """
    Extractor y validador de datos para el pipeline de recomendación.
    """
    
    def __init__(self, data_path: str, validation_rules: Optional[Dict] = None):
        """
        Inicializa el extractor de datos.
        
        Args:
            data_path: Ruta al directorio con los archivos de datos
            validation_rules: Reglas de validación para cada dataset
        """
        self.data_path = Path(data_path)
        self.validation_rules = validation_rules or self._get_default_validation_rules()
        
        # Rutas de los archivos
        self.transacciones_path = self.data_path / "transacciones.parquet"
        self.clientes_path = self.data_path / "clientes.parquet"
        self.productos_path = self.data_path / "productos.parquet"
        
        logger.info(f"DataExtractor inicializado con data_path: {self.data_path}")
    
    def _get_default_validation_rules(self) -> Dict:
        """Retorna las reglas de validación por defecto."""
        return {
            'transacciones': {
                'required_columns': ['customer_id', 'product_id', 'order_id', 'purchase_date', 'items'],
                'date_column': 'purchase_date',
                'min_records': 1000,
                'nullable_columns': []
            },
            'clientes': {
                'required_columns': ['customer_id', 'region_id', 'zone_id', 'customer_type'],
                'unique_key': 'customer_id',
                'min_records': 100
            },
            'productos': {
                'required_columns': ['product_id', 'brand', 'category', 'segment'],
                'unique_key': 'product_id',
                'min_records': 50
            }
        }
    
    def extract_and_validate(self) -> Dict[str, Any]:
        """
        Extrae y valida todos los datasets.
        
        Returns:
            Diccionario con estadísticas y metadatos de los datos
        """
        logger.info("Iniciando extracción y validación de datos...")
        
        results = {
            'extraction_timestamp': datetime.now().isoformat(),
            'data_path': str(self.data_path),
            'validation_passed': True,
            'errors': []
        }
        
        try:
            # Extraer transacciones
            transacciones_df, trans_stats = self._extract_transactions()
            results['transacciones'] = trans_stats
            
            # Extraer clientes  
            clientes_df, clientes_stats = self._extract_customers()
            results['clientes'] = clientes_stats
            
            # Extraer productos
            productos_df, productos_stats = self._extract_products()
            results['productos'] = productos_stats
            
            # Validaciones cruzadas
            cross_validation = self._cross_validate_datasets(
                transacciones_df, clientes_df, productos_df
            )
            results['cross_validation'] = cross_validation
            
            # Guardar datasets limpios
            self._save_clean_datasets(transacciones_df, clientes_df, productos_df)
            
            logger.info("✅ Extracción y validación completada exitosamente")
            
        except Exception as e:
            results['validation_passed'] = False
            results['errors'].append(str(e))
            logger.error(f"❌ Error en extracción de datos: {e}")
            raise
        
        return results
    
    def _extract_transactions(self) -> tuple[pd.DataFrame, Dict]:
        """Extrae y valida datos de transacciones."""
        logger.info("Extrayendo datos de transacciones...")
        
        if not self.transacciones_path.exists():
            raise FileNotFoundError(f"Archivo de transacciones no encontrado: {self.transacciones_path}")
        
        df = pd.read_parquet(self.transacciones_path)
        
        # SAMPLING: Usar solo 15,000 registros para entrenamiento rápido
        if len(df) > 15000:
            df = df.sample(n=15000, random_state=42)
            logger.info(f"📊 Dataset reducido a 15,000 registros para optimización")
        
        # Validaciones básicas
        self._validate_dataframe(df, 'transacciones')
        
        # Conversiones y limpieza
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        
        # Remover transacciones inválidas
        initial_rows = len(df)
        df = df.dropna(subset=['customer_id', 'product_id'])
        df = df[df['items'] != 0]  # Remover transacciones con 0 items
        
        logger.info(f"Transacciones procesadas: {len(df):,} ({initial_rows - len(df):,} removidas)")
        
        # Estadísticas
        stats = {
            'shape': df.shape,
            'date_range': {
                'start': df['purchase_date'].min().strftime('%Y-%m-%d'),
                'end': df['purchase_date'].max().strftime('%Y-%m-%d')
            },
            'unique_customers': df['customer_id'].nunique(),
            'unique_products': df['product_id'].nunique(),
            'unique_orders': df['order_id'].nunique(),
            'avg_items_per_transaction': df['items'].mean(),
            'records_removed': initial_rows - len(df)
        }
        
        return df, stats
    
    def _extract_customers(self) -> tuple[pd.DataFrame, Dict]:
        """Extrae y valida datos de clientes."""
        logger.info("Extrayendo datos de clientes...")
        
        if not self.clientes_path.exists():
            raise FileNotFoundError(f"Archivo de clientes no encontrado: {self.clientes_path}")
        
        df = pd.read_parquet(self.clientes_path)
        
        # Validaciones básicas
        self._validate_dataframe(df, 'clientes')
        
        # Remover duplicados por customer_id
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['customer_id'])
        
        logger.info(f"Clientes procesados: {len(df):,} ({initial_rows - len(df):,} duplicados removidos)")
        
        # Estadísticas
        stats = {
            'shape': df.shape,
            'unique_customers': df['customer_id'].nunique(),
            'regions': df['region_id'].nunique() if 'region_id' in df.columns else 0,
            'zones': df['zone_id'].nunique() if 'zone_id' in df.columns else 0,
            'customer_types': df['customer_type'].nunique() if 'customer_type' in df.columns else 0,
            'duplicates_removed': initial_rows - len(df)
        }
        
        return df, stats
    
    def _extract_products(self) -> tuple[pd.DataFrame, Dict]:
        """Extrae y valida datos de productos."""
        logger.info("Extrayendo datos de productos...")
        
        if not self.productos_path.exists():
            raise FileNotFoundError(f"Archivo de productos no encontrado: {self.productos_path}")
        
        df = pd.read_parquet(self.productos_path)
        
        # Validaciones básicas
        self._validate_dataframe(df, 'productos')
        
        # Remover duplicados por product_id
        initial_rows = len(df)
        df = df.drop_duplicates(subset=['product_id'])
        
        logger.info(f"Productos procesados: {len(df):,} ({initial_rows - len(df):,} duplicados removidos)")
        
        # Estadísticas
        stats = {
            'shape': df.shape,
            'unique_products': df['product_id'].nunique(),
            'brands': df['brand'].nunique() if 'brand' in df.columns else 0,
            'categories': df['category'].nunique() if 'category' in df.columns else 0,
            'segments': df['segment'].nunique() if 'segment' in df.columns else 0,
            'duplicates_removed': initial_rows - len(df)
        }
        
        return df, stats
    
    def _validate_dataframe(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Valida un dataframe según las reglas especificadas."""
        rules = self.validation_rules[dataset_name]
        
        # Verificar columnas requeridas
        required_cols = rules['required_columns']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en {dataset_name}: {missing_cols}")
        
        # Verificar número mínimo de registros
        min_records = rules.get('min_records', 0)
        if len(df) < min_records:
            raise ValueError(f"{dataset_name} tiene {len(df)} registros, mínimo requerido: {min_records}")
        
        # Verificar columna única si está especificada
        if 'unique_key' in rules:
            unique_col = rules['unique_key']
            duplicates = df[unique_col].duplicated().sum()
            if duplicates > 0:
                logger.warning(f"{duplicates} duplicados encontrados en {unique_col} de {dataset_name}")
        
        logger.info(f"Validación de {dataset_name} completada: {len(df):,} registros")
    
    def _cross_validate_datasets(self, transacciones_df: pd.DataFrame, 
                                clientes_df: pd.DataFrame, 
                                productos_df: pd.DataFrame) -> Dict:
        """Realiza validaciones cruzadas entre datasets."""
        logger.info("Realizando validaciones cruzadas...")
        
        results = {}
        
        # Verificar integridad referencial: clientes
        customer_ids_trans = set(transacciones_df['customer_id'].unique())
        customer_ids_master = set(clientes_df['customer_id'].unique())
        missing_customers = customer_ids_trans - customer_ids_master
        
        results['missing_customers'] = {
            'count': len(missing_customers),
            'percentage': len(missing_customers) / len(customer_ids_trans) * 100 if customer_ids_trans else 0
        }
        
        # Verificar integridad referencial: productos
        product_ids_trans = set(transacciones_df['product_id'].unique())
        product_ids_master = set(productos_df['product_id'].unique())
        missing_products = product_ids_trans - product_ids_master
        
        results['missing_products'] = {
            'count': len(missing_products),
            'percentage': len(missing_products) / len(product_ids_trans) * 100 if product_ids_trans else 0
        }
        
        # Estadísticas de cobertura
        results['coverage'] = {
            'customers_with_transactions': len(customer_ids_trans),
            'total_customers': len(customer_ids_master),
            'products_with_transactions': len(product_ids_trans),
            'total_products': len(product_ids_master)
        }
        
        if missing_customers:
            logger.warning(f"⚠️  {len(missing_customers)} clientes en transacciones no están en maestro de clientes")
        
        if missing_products:
            logger.warning(f"⚠️  {len(missing_products)} productos en transacciones no están en maestro de productos")
        
        logger.info(f"Validación cruzada completada")
        return results
    
    def _save_clean_datasets(self, transacciones_df: pd.DataFrame,
                           clientes_df: pd.DataFrame,
                           productos_df: pd.DataFrame) -> None:
        """Guarda los datasets limpios y validados."""
        logger.info("Guardando datasets limpios...")
        
        # Crear directorio de datos limpios si no existe
        clean_data_path = self.data_path / "clean"
        clean_data_path.mkdir(exist_ok=True)
        
        # Guardar datasets
        transacciones_df.to_parquet(clean_data_path / "transacciones_clean.parquet", index=False)
        clientes_df.to_parquet(clean_data_path / "clientes_clean.parquet", index=False)
        productos_df.to_parquet(clean_data_path / "productos_clean.parquet", index=False)
        
        # Guardar también como CSV para inspección manual
        transacciones_df.head(1000).to_csv(clean_data_path / "transacciones_sample.csv", index=False)
        clientes_df.to_csv(clean_data_path / "clientes_clean.csv", index=False)
        productos_df.to_csv(clean_data_path / "productos_clean.csv", index=False)
        
        logger.info(f"Datasets guardados en {clean_data_path}")

    def get_latest_week(self) -> str:
        """
        Retorna la semana más reciente en los datos de transacciones.
        
        Returns:
            String con la fecha de la semana más reciente
        """
        df = pd.read_parquet(self.transacciones_path)
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        latest_date = df['purchase_date'].max()
        
        return latest_date.strftime('%Y-%m-%d')

# Función de utilidad para uso independiente
def extract_sodai_data(data_path: str) -> Dict[str, Any]:
    """
    Función de conveniencia para extraer datos de SodAI.
    
    Args:
        data_path: Ruta al directorio con los archivos de datos
        
    Returns:
        Diccionario con estadísticas de extracción
    """
    extractor = DataExtractor(data_path)
    return extractor.extract_and_validate()

if __name__ == "__main__":
    # Ejemplo de uso
    import argparse
    
    parser = argparse.ArgumentParser(description='Extractor de datos SodAI Drinks')
    parser.add_argument('--data-path', required=True, help='Ruta al directorio de datos')
    args = parser.parse_args()
    
    result = extract_sodai_data(args.data_path)
    print("Extracción completada:")
    print(f"- Transacciones: {result['transacciones']['shape']}")
    print(f"- Clientes: {result['clientes']['shape']}")
    print(f"- Productos: {result['productos']['shape']}")
