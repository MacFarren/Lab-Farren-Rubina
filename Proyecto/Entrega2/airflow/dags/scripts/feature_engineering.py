"""
Módulo de Feature Engineering - SodAI Drinks Recommendation System
================================================================

Este módulo se encarga de crear features para el modelo de recomendación
basado en los datos históricos de transacciones, información de clientes
y catálogo de productos.

Features generadas:
- Frecuencia de compra por cliente
- Preferencias por categoría, marca y segmento
- Estacionalidad y patrones temporales
- Features de interacción cliente-producto
- Métricas de engagement y recencia

Autor: SodAI Drinks MLOps Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== TRANSFORMADORES PERSONALIZADOS ====================

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Trata valores atípicos usando capping basado en percentiles.
    Más robusto que eliminar outliers, mantiene todos los datos.
    """
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_cap = None
        self.upper_cap = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        self.lower_cap = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_cap = np.percentile(X, self.upper_percentile, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X_capped = X.copy()
        for i in range(X.shape[1]):
            X_capped[:, i] = np.clip(X_capped[:, i], self.lower_cap[i], self.upper_cap[i])
        return X_capped

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if self.n_features_in_ is None:
                raise RuntimeError("This transformer has not been fitted yet.")
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        else:
            return np.asarray(input_features, dtype=object)


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica transformación logarítmica a variables con distribución sesgada.
    Útil para normalizar distribuciones y reducir el efecto de outliers.
    """
    def __init__(self, offset=1):
        self.offset = offset  # Para evitar log(0)
        self.n_features_in_ = None

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.log(X + self.offset)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if self.n_features_in_ is None:
                raise RuntimeError("This transformer has not been fitted yet.")
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        else:
            return np.asarray(input_features, dtype=object)


# ========================================================================


class FeatureEngineer:
    """
    Generador de features para el sistema de recomendación.
    """
    
    def __init__(self, data_path: str):
        """
        Inicializa el feature engineer.
        
        Args:
            data_path: Ruta al directorio con los datos limpios
        """
        self.data_path = Path(data_path)
        self.clean_data_path = self.data_path / "clean"
        self.features_path = self.data_path / "features"
        
        # Crear directorio de features si no existe
        self.features_path.mkdir(exist_ok=True)
        
        logger.info(f"FeatureEngineer inicializado con data_path: {self.data_path}")
    
    def create_recommendation_features(self) -> Dict[str, Any]:
        """
        Crea el conjunto completo de features para recomendación.
        
        Returns:
            Diccionario con estadísticas del proceso
        """
        logger.info("Iniciando creación de features para recomendación...")
        
        # Cargar datasets limpios
        transacciones_df = pd.read_parquet(self.clean_data_path / "transacciones_clean.parquet")
        clientes_df = pd.read_parquet(self.clean_data_path / "clientes_clean.parquet")
        productos_df = pd.read_parquet(self.clean_data_path / "productos_clean.parquet")
        
        logger.info(f"Datasets cargados: {transacciones_df.shape[0]:,} transacciones, {clientes_df.shape[0]:,} clientes, {productos_df.shape[0]:,} productos")
        
        # 1. Features de cliente
        customer_features = self._create_customer_features(transacciones_df, clientes_df)
        
        # 2. Features de producto
        product_features = self._create_product_features(transacciones_df, productos_df)
        
        # 3. Features de interacción cliente-producto
        interaction_features = self._create_interaction_features(transacciones_df)
        
        # 4. Features temporales
        temporal_features = self._create_temporal_features(transacciones_df)
        
        # 5. Crear dataset final para entrenamiento
        final_dataset = self._build_training_dataset(
            transacciones_df, customer_features, product_features, 
            interaction_features, temporal_features, clientes_df, productos_df
        )
        
        # 6. Aplicar pipeline de preprocessing robusto
        logger.info("Aplicando pipeline de preprocessing con OutlierCapper y LogTransformer...")
        final_dataset_processed = self._apply_preprocessing_pipeline(final_dataset)
        
        # Guardar features y dataset
        self._save_features(customer_features, product_features, interaction_features, 
                           temporal_features, final_dataset_processed)
        
        # Estadísticas
        stats = {
            'customer_features_shape': customer_features.shape,
            'product_features_shape': product_features.shape,
            'interaction_features_shape': interaction_features.shape,
            'temporal_features_shape': temporal_features.shape,
            'final_dataset_shape': final_dataset_processed.shape,
            'num_features': final_dataset_processed.shape[1] - 3,  # Excluyendo customer_id, product_id, target
            'customer_product_pairs': final_dataset_processed.shape[0],
            'positive_interactions': final_dataset_processed['target'].sum(),
            'negative_interactions': (final_dataset_processed['target'] == 0).sum(),
            'preprocessing_applied': True,
            'pipeline_saved': True
        }
        
        logger.info(f"✅ Features creados exitosamente: {final_dataset_processed.shape}")
        return stats
    
    def _create_customer_features(self, transacciones_df: pd.DataFrame, 
                                clientes_df: pd.DataFrame) -> pd.DataFrame:
        """Crea features a nivel de cliente."""
        logger.info("Creando features de cliente...")
        
        # Features básicos de actividad
        customer_activity = transacciones_df.groupby('customer_id').agg({
            'order_id': 'nunique',  # Número de órdenes
            'product_id': ['nunique', 'count'],  # Productos únicos y total de transacciones
            'items': ['sum', 'mean', 'std'],  # Items totales, promedio y std
            'purchase_date': ['min', 'max', 'nunique']  # Primera compra, última compra, días únicos
        }).reset_index()
        
        # Aplanar columnas multi-nivel
        customer_activity.columns = ['customer_id'] + [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
                                                      for col in customer_activity.columns[1:]]
        
        # Renombrar columnas para claridad
        customer_activity.rename(columns={
            'order_id_nunique': 'total_orders',
            'product_id_nunique': 'unique_products_bought',
            'product_id_count': 'total_transactions',
            'items_sum': 'total_items',
            'items_mean': 'avg_items_per_transaction',
            'items_std': 'std_items_per_transaction',
            'purchase_date_min': 'first_purchase',
            'purchase_date_max': 'last_purchase',
            'purchase_date_nunique': 'active_days'
        }, inplace=True)
        
        # Calcular features derivados
        customer_activity['first_purchase'] = pd.to_datetime(customer_activity['first_purchase'])
        customer_activity['last_purchase'] = pd.to_datetime(customer_activity['last_purchase'])
        reference_date = transacciones_df['purchase_date'].max()
        
        customer_activity['days_since_first_purchase'] = (reference_date - customer_activity['first_purchase']).dt.days
        customer_activity['days_since_last_purchase'] = (reference_date - customer_activity['last_purchase']).dt.days
        customer_activity['customer_lifetime_days'] = (customer_activity['last_purchase'] - customer_activity['first_purchase']).dt.days + 1
        customer_activity['purchase_frequency'] = customer_activity['total_orders'] / customer_activity['customer_lifetime_days']
        customer_activity['avg_items_per_order'] = customer_activity['total_items'] / customer_activity['total_orders']
        customer_activity['product_diversity'] = customer_activity['unique_products_bought'] / customer_activity['total_transactions']
        
        # Llenar NaNs
        customer_activity['std_items_per_transaction'].fillna(0, inplace=True)
        customer_activity['purchase_frequency'].fillna(0, inplace=True)
        
        # Features de preferencias por categoría
        category_prefs = self._calculate_category_preferences(transacciones_df)
        
        # Unir con información demográfica de clientes
        customer_features = customer_activity.merge(clientes_df, on='customer_id', how='left')
        customer_features = customer_features.merge(category_prefs, on='customer_id', how='left')
        
        # Crear features adicionales de clientes
        if 'X' in customer_features.columns and 'Y' in customer_features.columns:
            customer_features['location_distance_from_origin'] = np.sqrt(
                customer_features['X']**2 + customer_features['Y']**2
            )
        
        # Encoding de variables categóricas
        categorical_cols = ['customer_type', 'region_id', 'zone_id']
        for col in categorical_cols:
            if col in customer_features.columns:
                le = LabelEncoder()
                customer_features[f'{col}_encoded'] = le.fit_transform(customer_features[col].fillna('unknown'))
        
        logger.info(f"Features de cliente creados: {customer_features.shape}")
        return customer_features
    
    def _calculate_category_preferences(self, transacciones_df: pd.DataFrame) -> pd.DataFrame:
        """Calcula preferencias del cliente por categorías de productos."""
        # Cargar datos de productos para obtener categorías
        productos_df = pd.read_parquet(self.clean_data_path / "productos_clean.parquet")
        
        # Unir transacciones con información de productos
        trans_with_products = transacciones_df.merge(productos_df[['product_id', 'category', 'brand', 'segment']], 
                                                   on='product_id', how='left')
        
        # Preferencias por categoría
        category_prefs = trans_with_products.groupby(['customer_id', 'category']).agg({
            'items': 'sum',
            'order_id': 'nunique'
        }).reset_index()
        
        # Crear features de preferencia por categoría top
        top_categories = trans_with_products['category'].value_counts().head(5).index.tolist()
        
        customer_category_prefs = pd.DataFrame()
        for category in top_categories:
            cat_data = category_prefs[category_prefs['category'] == category][['customer_id', 'items', 'order_id']]
            cat_data.columns = ['customer_id', f'items_{category.lower().replace(" ", "_")}', 
                              f'orders_{category.lower().replace(" ", "_")}']
            
            if customer_category_prefs.empty:
                customer_category_prefs = cat_data
            else:
                customer_category_prefs = customer_category_prefs.merge(cat_data, on='customer_id', how='outer')
        
        # Llenar NaNs con 0
        customer_category_prefs.fillna(0, inplace=True)
        
        # Agregar número de categorías compradas por cliente
        categories_per_customer = trans_with_products.groupby('customer_id')['category'].nunique().reset_index()
        categories_per_customer.rename(columns={'category': 'unique_categories_bought'}, inplace=True)
        
        customer_category_prefs = customer_category_prefs.merge(categories_per_customer, on='customer_id', how='left')
        
        return customer_category_prefs
    
    def _create_product_features(self, transacciones_df: pd.DataFrame, 
                               productos_df: pd.DataFrame) -> pd.DataFrame:
        """Crea features a nivel de producto."""
        logger.info("Creando features de producto...")
        
        # Features básicos de popularidad del producto
        product_activity = transacciones_df.groupby('product_id').agg({
            'customer_id': 'nunique',  # Número de clientes únicos
            'order_id': 'nunique',     # Número de órdenes
            'items': ['sum', 'mean', 'std'],  # Total, promedio y std de items
            'purchase_date': ['min', 'max', 'nunique']  # Primera venta, última venta, días únicos
        }).reset_index()
        
        # Aplanar columnas
        product_activity.columns = ['product_id'] + [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
                                                    for col in product_activity.columns[1:]]
        
        # Renombrar columnas
        product_activity.rename(columns={
            'customer_id_nunique': 'unique_customers',
            'order_id_nunique': 'total_orders',
            'items_sum': 'total_items_sold',
            'items_mean': 'avg_items_per_transaction',
            'items_std': 'std_items_per_transaction',
            'purchase_date_min': 'first_sale',
            'purchase_date_max': 'last_sale',
            'purchase_date_nunique': 'active_sale_days'
        }, inplace=True)
        
        # Features derivados
        reference_date = transacciones_df['purchase_date'].max()
        product_activity['first_sale'] = pd.to_datetime(product_activity['first_sale'])
        product_activity['last_sale'] = pd.to_datetime(product_activity['last_sale'])
        
        product_activity['days_since_first_sale'] = (reference_date - product_activity['first_sale']).dt.days
        product_activity['days_since_last_sale'] = (reference_date - product_activity['last_sale']).dt.days
        product_activity['product_lifetime_days'] = (product_activity['last_sale'] - product_activity['first_sale']).dt.days + 1
        product_activity['sales_frequency'] = product_activity['total_orders'] / product_activity['product_lifetime_days']
        product_activity['customer_penetration'] = product_activity['unique_customers'] / transacciones_df['customer_id'].nunique()
        
        # Llenar NaNs
        product_activity['std_items_per_transaction'].fillna(0, inplace=True)
        product_activity['sales_frequency'].fillna(0, inplace=True)
        
        # Unir con información de productos
        product_features = product_activity.merge(productos_df, on='product_id', how='left')
        
        # Encoding de variables categóricas de productos
        categorical_cols = ['brand', 'category', 'sub_category', 'segment', 'package']
        for col in categorical_cols:
            if col in product_features.columns:
                le = LabelEncoder()
                product_features[f'{col}_encoded'] = le.fit_transform(product_features[col].fillna('unknown'))
        
        # Features de tamaño si existe
        if 'size' in product_features.columns:
            product_features['size_numeric'] = pd.to_numeric(product_features['size'], errors='coerce')
            product_features['size_numeric'].fillna(product_features['size_numeric'].median(), inplace=True)
        
        logger.info(f"Features de producto creados: {product_features.shape}")
        return product_features
    
    def _create_interaction_features(self, transacciones_df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de interacción cliente-producto."""
        logger.info("Creando features de interacción...")
        
        # Features básicos de interacción
        interaction_features = transacciones_df.groupby(['customer_id', 'product_id']).agg({
            'items': ['sum', 'count', 'mean', 'std'],
            'order_id': 'nunique',
            'purchase_date': ['min', 'max', 'nunique']
        }).reset_index()
        
        # Aplanar columnas
        interaction_features.columns = ['customer_id', 'product_id'] + \
                                     [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
                                      for col in interaction_features.columns[2:]]
        
        # Renombrar columnas
        interaction_features.rename(columns={
            'items_sum': 'total_items_bought',
            'items_count': 'interaction_frequency',
            'items_mean': 'avg_items_per_interaction',
            'items_std': 'std_items_per_interaction',
            'order_id_nunique': 'unique_orders',
            'purchase_date_min': 'first_interaction',
            'purchase_date_max': 'last_interaction',
            'purchase_date_nunique': 'interaction_days'
        }, inplace=True)
        
        # Features temporales de interacción
        reference_date = transacciones_df['purchase_date'].max()
        interaction_features['first_interaction'] = pd.to_datetime(interaction_features['first_interaction'])
        interaction_features['last_interaction'] = pd.to_datetime(interaction_features['last_interaction'])
        
        interaction_features['days_since_first_interaction'] = (reference_date - interaction_features['first_interaction']).dt.days
        interaction_features['days_since_last_interaction'] = (reference_date - interaction_features['last_interaction']).dt.days
        interaction_features['interaction_span_days'] = (interaction_features['last_interaction'] - interaction_features['first_interaction']).dt.days + 1
        interaction_features['avg_days_between_interactions'] = interaction_features['interaction_span_days'] / interaction_features['interaction_frequency']
        
        # Features de lealtad/preferencia
        interaction_features['loyalty_score'] = interaction_features['interaction_frequency'] * interaction_features['total_items_bought']
        interaction_features['recency_score'] = 1 / (interaction_features['days_since_last_interaction'] + 1)
        
        # Llenar NaNs
        interaction_features['std_items_per_interaction'].fillna(0, inplace=True)
        interaction_features['avg_days_between_interactions'].fillna(0, inplace=True)
        
        logger.info(f"Features de interacción creados: {interaction_features.shape}")
        return interaction_features
    
    def _create_temporal_features(self, transacciones_df: pd.DataFrame) -> pd.DataFrame:
        """Crea features temporales basados en estacionalidad y tendencias."""
        logger.info("Creando features temporales...")
        
        df = transacciones_df.copy()
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        
        # Extraer componentes temporales
        df['day_of_week'] = df['purchase_date'].dt.dayofweek
        df['month'] = df['purchase_date'].dt.month
        df['quarter'] = df['purchase_date'].dt.quarter
        df['week_of_year'] = df['purchase_date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Features temporales por cliente-producto
        temporal_features = df.groupby(['customer_id', 'product_id']).agg({
            'day_of_week': lambda x: x.value_counts().index[0],  # Día de la semana más común
            'month': lambda x: x.value_counts().index[0],        # Mes más común
            'is_weekend': 'mean',                                # Proporción de compras en weekend
            'week_of_year': ['min', 'max', 'nunique']            # Estacionalidad
        }).reset_index()
        
        # Aplanar columnas
        temporal_features.columns = ['customer_id', 'product_id', 
                                   'preferred_day_of_week', 'preferred_month', 'weekend_purchase_ratio'] + \
                                  [f"week_{col[1]}" for col in temporal_features.columns[5:]]
        
        # Features de estacionalidad
        temporal_features['seasonal_span_weeks'] = temporal_features['week_max'] - temporal_features['week_min'] + 1
        temporal_features['is_seasonal_buyer'] = (temporal_features['week_nunique'] < 10).astype(int)
        
        logger.info(f"Features temporales creados: {temporal_features.shape}")
        return temporal_features
    
    def _build_training_dataset(self, transacciones_df: pd.DataFrame,
                              customer_features: pd.DataFrame,
                              product_features: pd.DataFrame,
                              interaction_features: pd.DataFrame,
                              temporal_features: pd.DataFrame,
                              clientes_df: pd.DataFrame,
                              productos_df: pd.DataFrame) -> pd.DataFrame:
        """Construye el dataset final para entrenamiento con interacciones positivas y negativas."""
        logger.info("Construyendo dataset final para entrenamiento...")
        
        # Crear interacciones positivas (1 = compró el producto)
        positive_interactions = interaction_features[['customer_id', 'product_id']].copy()
        positive_interactions['target'] = 1
        
        # Generar interacciones negativas (0 = no compró el producto)
        # Usar un sampling estratégico para balancear el dataset
        all_customers = set(clientes_df['customer_id'].unique())
        all_products = set(productos_df['product_id'].unique())
        existing_interactions = set(zip(transacciones_df['customer_id'], transacciones_df['product_id']))
        
        # Generar muestras negativas
        negative_samples = []
        target_negative_samples = len(positive_interactions) * 2  # Ratio 1:2 positivo:negativo
        
        import random
        random.seed(42)
        
        customers_list = list(all_customers)
        products_list = list(all_products)
        
        while len(negative_samples) < target_negative_samples:
            customer_id = random.choice(customers_list)
            product_id = random.choice(products_list)
            
            if (customer_id, product_id) not in existing_interactions:
                negative_samples.append({'customer_id': customer_id, 'product_id': product_id, 'target': 0})
                existing_interactions.add((customer_id, product_id))
        
        negative_interactions = pd.DataFrame(negative_samples)
        
        # Combinar interacciones positivas y negativas
        all_interactions = pd.concat([positive_interactions, negative_interactions], ignore_index=True)
        
        # Unir con features
        dataset = all_interactions.merge(
            customer_features.drop(['first_purchase', 'last_purchase'], axis=1, errors='ignore'), 
            on='customer_id', how='left'
        )
        
        dataset = dataset.merge(
            product_features.drop(['first_sale', 'last_sale'], axis=1, errors='ignore'),
            on='product_id', how='left'
        )
        
        # Para interacciones negativas, agregar valores por defecto en features de interacción
        interaction_features_filled = interaction_features.copy()
        
        dataset = dataset.merge(interaction_features_filled, on=['customer_id', 'product_id'], how='left')
        dataset = dataset.merge(temporal_features, on=['customer_id', 'product_id'], how='left')
        
        # Llenar NaNs en features de interacción para muestras negativas
        interaction_columns = [col for col in interaction_features.columns if col not in ['customer_id', 'product_id']]
        temporal_columns = [col for col in temporal_features.columns if col not in ['customer_id', 'product_id']]
        
        for col in interaction_columns + temporal_columns:
            if col in dataset.columns:
                # Usar pandas.api.types para verificar tipos numéricos de manera más robusta
                if pd.api.types.is_numeric_dtype(dataset[col]):
                    dataset[col] = dataset[col].fillna(0)
                else:
                    dataset[col] = dataset[col].fillna('unknown')
        
        # Reordenar columnas
        id_columns = ['customer_id', 'product_id', 'target']
        feature_columns = [col for col in dataset.columns if col not in id_columns]
        dataset = dataset[id_columns + feature_columns]
        
        # Remover filas con demasiados NaNs
        dataset.dropna(thresh=len(dataset.columns) * 0.8, inplace=True)
        
        logger.info(f"Dataset final creado: {dataset.shape}")
        logger.info(f"Interacciones positivas: {(dataset['target'] == 1).sum():,}")
        logger.info(f"Interacciones negativas: {(dataset['target'] == 0).sum():,}")
        
        return dataset
    
    def _apply_preprocessing_pipeline(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica pipeline de preprocessing robusto con OutlierCapper y LogTransformer.
        
        Justificación de cada paso:
        1. IMPUTACIÓN: Mediana para numéricas (robusta), moda para categóricas
        2. OUTLIER CAPPING: Percentiles 1-99, mantiene 98% de datos
        3. LOG TRANSFORM: Para variables con distribución sesgada
        4. ESCALADO: RobustScaler usa mediana/IQR, más robusto que StandardScaler
        5. ONE-HOT ENCODING: Para variables nominales sin orden
        """
        logger.info("📊 Aplicando pipeline de preprocessing robusto...")
        
        # Identificar columnas por tipo
        id_cols = ['customer_id', 'product_id']
        target_col = 'target'
        
        # ELIMINAR columnas timestamp que causan error en OneHotEncoder
        timestamp_cols = ['first_interaction', 'last_interaction']
        for col in timestamp_cols:
            if col in dataset.columns:
                dataset = dataset.drop(columns=[col])
                logger.info(f' Columna timestamp eliminada: {col}')
        
        # Separar features de IDs y target
        feature_cols = [col for col in dataset.columns if col not in id_cols + [target_col]]
        
        # Identificar tipos de columnas en los features
        numeric_cols = []
        categorical_cols = []
        log_transform_cols = []
        
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(dataset[col]):
                # Variables que se benefician de transformación log (distribuciones sesgadas)
                if col in ['num_deliver_per_week', 'num_visit_per_week', 'total_orders', 
                          'total_transactions', 'total_items', 'days_since_first_purchase']:
                    log_transform_cols.append(col)
                else:
                    numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        logger.info(f"Columnas detectadas: {len(numeric_cols)} numéricas estándar, {len(log_transform_cols)} log-transform, {len(categorical_cols)} categóricas")
        
        # Crear pipelines por tipo
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier_capper', OutlierCapper(lower_percentile=1, upper_percentile=99)),
            ('scaler', RobustScaler())
        ])
        
        log_numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('log_transform', LogTransformer(offset=1)),
            ('outlier_capper', OutlierCapper(lower_percentile=1, upper_percentile=99)),
            ('scaler', RobustScaler())
        ])
        
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])
        
        # ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num_standard', numeric_pipeline, numeric_cols),
                ('num_log', log_numeric_pipeline, log_transform_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        # Separar features de IDs/target
        X = dataset[feature_cols]
        
        # Ajustar y transformar
        X_transformed = preprocessor.fit_transform(X)
        
        # Obtener nombres de features transformadas
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            # Fallback si falla get_feature_names_out
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        
        # Crear DataFrame con features transformadas
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=dataset.index)
        
        # Recombinar con IDs y target
        final_df = pd.concat([
            dataset[id_cols + [target_col]].reset_index(drop=True),
            X_transformed_df.reset_index(drop=True)
        ], axis=1)
        
        # Guardar el pipeline para uso futuro
        joblib.dump(preprocessor, self.features_path / 'preprocessing_pipeline.pkl')
        logger.info(f"✅ Pipeline guardado en {self.features_path / 'preprocessing_pipeline.pkl'}")
        
        logger.info(f"✅ Preprocessing completado: {final_df.shape}")
        logger.info(f"   - Features originales: {len(feature_cols)}")
        logger.info(f"   - Features transformadas: {X_transformed.shape[1]}")
        logger.info(f"   - Nulos restantes: {final_df.isnull().sum().sum()}")
        
        return final_df
    
    def _save_features(self, customer_features: pd.DataFrame,
                      product_features: pd.DataFrame,
                      interaction_features: pd.DataFrame,
                      temporal_features: pd.DataFrame,
                      final_dataset: pd.DataFrame) -> None:
        """Guarda todos los features generados."""
        logger.info("Guardando features...")
        
        # Limpiar tipos de datos antes de guardar
        def clean_datatypes_for_parquet(df):
            """Limpia tipos de datos para ser compatibles con parquet."""
            df = df.copy()
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Convertir objetos a string si contienen datos mixtos
                    df[col] = df[col].astype(str)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    # Convertir tipos numéricos que pueden tener problemas
                    if 'int' in str(df[col].dtype) and df[col].isna().any():
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    elif df[col].dtype == 'UInt32':
                        df[col] = df[col].astype('int64')
            return df
        
        # Limpiar dataframes antes de guardar
        customer_features = clean_datatypes_for_parquet(customer_features)
        product_features = clean_datatypes_for_parquet(product_features)
        interaction_features = clean_datatypes_for_parquet(interaction_features)
        temporal_features = clean_datatypes_for_parquet(temporal_features)
        final_dataset = clean_datatypes_for_parquet(final_dataset)
        
        # Guardar features individuales
        customer_features.to_parquet(self.features_path / "customer_features.parquet", index=False)
        product_features.to_parquet(self.features_path / "product_features.parquet", index=False)
        interaction_features.to_parquet(self.features_path / "interaction_features.parquet", index=False)
        temporal_features.to_parquet(self.features_path / "temporal_features.parquet", index=False)
        
        # Guardar dataset final
        final_dataset.to_parquet(self.features_path / "training_dataset.parquet", index=False)
        
        # Guardar también una muestra como CSV para inspección
        final_dataset.head(10000).to_csv(self.features_path / "training_dataset_sample.csv", index=False)
        
        # Guardar metadatos de features
        feature_metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'num_features': final_dataset.shape[1] - 3,
            'total_samples': final_dataset.shape[0],
            'feature_columns': [col for col in final_dataset.columns if col not in ['customer_id', 'product_id', 'target']]
        }
        
        import json
        with open(self.features_path / "feature_metadata.json", 'w') as f:
            json.dump(feature_metadata, f, indent=2)
        
        logger.info(f"Features guardados en {self.features_path}")

# Función de utilidad para uso independiente
def create_sodai_features(data_path: str) -> Dict[str, Any]:
    """
    Función de conveniencia para crear features de SodAI.
    
    Args:
        data_path: Ruta al directorio con los datos
        
    Returns:
        Diccionario con estadísticas de feature engineering
    """
    engineer = FeatureEngineer(data_path)
    return engineer.create_recommendation_features()

if __name__ == "__main__":
    # Ejemplo de uso
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Engineer para SodAI Drinks')
    parser.add_argument('--data-path', required=True, help='Ruta al directorio de datos')
    args = parser.parse_args()
    
    result = create_sodai_features(args.data_path)
    print("Features creados exitosamente:")
    print(f"- Dataset final: {result['final_dataset_shape']}")
    print(f"- Número de features: {result['num_features']}")
    print(f"- Pares cliente-producto: {result['customer_product_pairs']:,}")

