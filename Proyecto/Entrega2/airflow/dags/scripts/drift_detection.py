"""
Módulo de Detección de Drift - SodAI Drinks Recommendation System
===============================================================

Este módulo implementa detección de drift estadístico para determinar
cuándo los datos han cambiado significativamente y se requiere 
reentrenamiento del modelo.

Métodos de detección:
- Kolmogorov-Smirnov test para variables numéricas
- Chi-cuadrado test para variables categóricas
- Population Stability Index (PSI)
- Tests de distribución específicos

Autor: SodAI Drinks MLOps Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import warnings

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Detector de drift de datos para el pipeline de recomendación.
    """
    
    def __init__(self, reference_data_path: str, current_data_path: str,
                 drift_threshold: float = 0.1):
        """
        Inicializa el detector de drift.
        
        Args:
            reference_data_path: Ruta a los datos de referencia
            current_data_path: Ruta a los datos actuales
            drift_threshold: Umbral para considerar drift significativo
        """
        self.reference_data_path = Path(reference_data_path)
        self.current_data_path = Path(current_data_path)
        self.drift_threshold = drift_threshold
        
        logger.info(f"DriftDetector inicializado con threshold: {drift_threshold}")
    
    def detect_drift(self) -> Dict[str, Any]:
        """
        Ejecuta la detección de drift completa.
        
        Returns:
            Diccionario con resultados de detección de drift
        """
        logger.info("Iniciando detección de drift...")
        
        try:
            # Preparar datos para comparación
            reference_data, current_data = self._prepare_data_for_comparison()
            
            # Ejecutar tests de drift
            drift_results = {
                'detection_timestamp': datetime.now().isoformat(),
                'drift_threshold': self.drift_threshold,
                'tests_performed': []
            }
            
            # 1. Tests estadísticos por feature
            statistical_tests = self._run_statistical_tests(reference_data, current_data)
            drift_results['statistical_tests'] = statistical_tests
            drift_results['tests_performed'].extend(['ks_test', 'chi2_test'])
            
            # 2. Population Stability Index (PSI)
            psi_results = self._calculate_psi(reference_data, current_data)
            drift_results['psi_results'] = psi_results
            drift_results['tests_performed'].append('psi')
            
            # 3. Tests de distribución
            distribution_tests = self._run_distribution_tests(reference_data, current_data)
            drift_results['distribution_tests'] = distribution_tests
            drift_results['tests_performed'].append('distribution_tests')
            
            # 4. Análisis de features más importantes
            feature_importance_drift = self._analyze_feature_importance_drift(reference_data, current_data)
            drift_results['feature_importance_drift'] = feature_importance_drift
            
            # 5. Decidir si hay drift significativo
            drift_decision = self._make_drift_decision(drift_results)
            drift_results.update(drift_decision)
            
            logger.info(f"Detección de drift completada. Drift detectado: {drift_results['drift_detected']}")
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Error en detección de drift: {e}")
            # En caso de error, asumir que hay drift para ser conservador
            return {
                'detection_timestamp': datetime.now().isoformat(),
                'drift_detected': True,
                'drift_score': 1.0,
                'error': str(e),
                'message': 'Error en detección - asumiendo drift por seguridad'
            }
    
    def _prepare_data_for_comparison(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepara los datos de referencia y actuales para comparación."""
        logger.info("Preparando datos para comparación...")
        
        # Si no existe archivo de referencia, usar datos históricos
        if not self.reference_data_path.exists():
            logger.warning("No hay datos de referencia. Creando baseline...")
            reference_data = self._create_reference_baseline()
        else:
            reference_data = pd.read_parquet(self.reference_data_path)
        
        # Cargar datos actuales
        if not self.current_data_path.exists():
            # Si no hay datos actuales específicos, usar los features más recientes
            current_data = self._extract_current_features()
        else:
            current_data = pd.read_parquet(self.current_data_path)
        
        # Asegurar que ambos datasets tienen las mismas columnas
        common_columns = set(reference_data.columns).intersection(set(current_data.columns))
        reference_data = reference_data[list(common_columns)]
        current_data = current_data[list(common_columns)]
        
        logger.info(f"Datos preparados: {len(reference_data):,} referencias vs {len(current_data):,} actuales")
        logger.info(f"Features comparados: {len(common_columns)}")
        
        return reference_data, current_data
    
    def _create_reference_baseline(self) -> pd.DataFrame:
        """Crea baseline de referencia desde los datos existentes."""
        features_path = Path(self.reference_data_path).parent / "features"
        training_data_path = features_path / "training_dataset.parquet"
        
        if training_data_path.exists():
            full_data = pd.read_parquet(training_data_path)
            # Tomar el 70% más antiguo como referencia
            sample_size = int(len(full_data) * 0.7)
            reference_data = full_data.head(sample_size)
            
            # Guardar como referencia para futuras comparaciones
            reference_data.to_parquet(self.reference_data_path, index=False)
            logger.info(f"Baseline de referencia creado: {len(reference_data):,} muestras")
            
            return reference_data
        else:
            raise FileNotFoundError("No se pueden encontrar datos para crear baseline de referencia")
    
    def _extract_current_features(self) -> pd.DataFrame:
        """Extrae features actuales para comparación."""
        features_path = Path(self.current_data_path).parent / "features"
        training_data_path = features_path / "training_dataset.parquet"
        
        if training_data_path.exists():
            full_data = pd.read_parquet(training_data_path)
            # Tomar el 30% más reciente como datos actuales
            sample_size = int(len(full_data) * 0.3)
            current_data = full_data.tail(sample_size)
            
            # Guardar para futuras comparaciones
            current_data.to_parquet(self.current_data_path, index=False)
            logger.info(f"Datos actuales extraídos: {len(current_data):,} muestras")
            
            return current_data
        else:
            raise FileNotFoundError("No se pueden encontrar datos actuales")
    
    def _run_statistical_tests(self, reference_data: pd.DataFrame, 
                             current_data: pd.DataFrame) -> Dict[str, Any]:
        """Ejecuta tests estadísticos para detección de drift."""
        logger.info("Ejecutando tests estadísticos...")
        
        results = {
            'ks_test_results': {},
            'chi2_test_results': {},
            'significant_drifts': [],
            'p_value_threshold': 0.05
        }
        
        # Identificar tipos de columnas
        numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
        categorical_columns = reference_data.select_dtypes(include=['object', 'category']).columns
        
        # Kolmogorov-Smirnov test para variables numéricas
        for col in numeric_columns:
            if col in ['customer_id', 'product_id', 'target']:
                continue
                
            try:
                # Remover NaNs y valores infinitos
                ref_values = reference_data[col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
                curr_values = current_data[col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(ref_values) > 50 and len(curr_values) > 50:
                    ks_stat, p_value = ks_2samp(ref_values, curr_values)
                    
                    results['ks_test_results'][col] = {
                        'ks_statistic': float(ks_stat),
                        'p_value': float(p_value),
                        'drift_detected': p_value < results['p_value_threshold'],
                        'effect_size': float(ks_stat)  # KS statistic es también una medida de effect size
                    }
                    
                    if p_value < results['p_value_threshold']:
                        results['significant_drifts'].append({
                            'feature': col,
                            'test_type': 'ks_test',
                            'p_value': float(p_value),
                            'effect_size': float(ks_stat)
                        })
                        
            except Exception as e:
                logger.warning(f"Error en KS test para {col}: {e}")
                continue
        
        # Chi-cuadrado test para variables categóricas
        for col in categorical_columns:
            if col in ['customer_id', 'product_id']:
                continue
                
            try:
                # Crear tabla de contingencia
                ref_counts = reference_data[col].value_counts()
                curr_counts = current_data[col].value_counts()
                
                # Obtener categorías comunes
                common_categories = set(ref_counts.index).intersection(set(curr_counts.index))
                
                if len(common_categories) > 1:
                    ref_common = ref_counts.reindex(common_categories, fill_value=0)
                    curr_common = curr_counts.reindex(common_categories, fill_value=0)
                    
                    # Crear matriz para chi2 test
                    contingency_table = np.array([ref_common.values, curr_common.values])
                    
                    if contingency_table.sum() > 0:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        # Calcular Cramér's V como medida de effect size
                        n = contingency_table.sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                        
                        results['chi2_test_results'][col] = {
                            'chi2_statistic': float(chi2),
                            'p_value': float(p_value),
                            'degrees_of_freedom': int(dof),
                            'cramers_v': float(cramers_v),
                            'drift_detected': p_value < results['p_value_threshold']
                        }
                        
                        if p_value < results['p_value_threshold']:
                            results['significant_drifts'].append({
                                'feature': col,
                                'test_type': 'chi2_test',
                                'p_value': float(p_value),
                                'effect_size': float(cramers_v)
                            })
                            
            except Exception as e:
                logger.warning(f"Error en Chi2 test para {col}: {e}")
                continue
        
        logger.info(f"Tests estadísticos completados. Drifts significativos: {len(results['significant_drifts'])}")
        return results
    
    def _calculate_psi(self, reference_data: pd.DataFrame, 
                      current_data: pd.DataFrame) -> Dict[str, Any]:
        """Calcula Population Stability Index (PSI) para features numéricas."""
        logger.info("Calculando Population Stability Index (PSI)...")
        
        psi_results = {
            'psi_scores': {},
            'psi_threshold_low': 0.1,
            'psi_threshold_medium': 0.2,
            'psi_threshold_high': 0.25,
            'features_with_drift': []
        }
        
        numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['customer_id', 'product_id', 'target']:
                continue
                
            try:
                ref_values = reference_data[col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
                curr_values = current_data[col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(ref_values) > 100 and len(curr_values) > 100:
                    psi_score = self._calculate_psi_for_feature(ref_values, curr_values)
                    
                    # Clasificar nivel de drift
                    if psi_score < psi_results['psi_threshold_low']:
                        drift_level = 'no_drift'
                    elif psi_score < psi_results['psi_threshold_medium']:
                        drift_level = 'low_drift'
                    elif psi_score < psi_results['psi_threshold_high']:
                        drift_level = 'medium_drift'
                    else:
                        drift_level = 'high_drift'
                    
                    psi_results['psi_scores'][col] = {
                        'psi_score': float(psi_score),
                        'drift_level': drift_level,
                        'drift_detected': drift_level != 'no_drift'
                    }
                    
                    if drift_level != 'no_drift':
                        psi_results['features_with_drift'].append({
                            'feature': col,
                            'psi_score': float(psi_score),
                            'drift_level': drift_level
                        })
                        
            except Exception as e:
                logger.warning(f"Error calculando PSI para {col}: {e}")
                continue
        
        logger.info(f"PSI calculado para {len(psi_results['psi_scores'])} features")
        return psi_results
    
    def _calculate_psi_for_feature(self, reference: np.ndarray, current: np.ndarray, 
                                 buckets: int = 10) -> float:
        """Calcula PSI para una feature específica."""
        # Crear bins basados en los percentiles de referencia
        breakpoints = np.linspace(0, 100, buckets + 1)
        percentiles = np.percentile(reference, breakpoints)
        percentiles = np.unique(percentiles)  # Remover duplicados
        
        if len(percentiles) < 3:
            return 0.0  # No se puede calcular PSI con muy pocos bins
        
        # Calcular distribuciones
        ref_counts, _ = np.histogram(reference, bins=percentiles)
        curr_counts, _ = np.histogram(current, bins=percentiles)
        
        # Normalizar a probabilidades
        ref_props = ref_counts / ref_counts.sum()
        curr_props = curr_counts / curr_counts.sum()
        
        # Agregar pequeña constante para evitar log(0)
        epsilon = 1e-6
        ref_props = ref_props + epsilon
        curr_props = curr_props + epsilon
        
        # Calcular PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        
        return psi
    
    def _run_distribution_tests(self, reference_data: pd.DataFrame, 
                              current_data: pd.DataFrame) -> Dict[str, Any]:
        """Ejecuta tests específicos de distribución."""
        logger.info("Ejecutando tests de distribución...")
        
        results = {
            'mean_shift_tests': {},
            'variance_tests': {},
            'normality_tests': {},
            'significant_changes': []
        }
        
        numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['customer_id', 'product_id', 'target']:
                continue
                
            try:
                ref_values = reference_data[col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
                curr_values = current_data[col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(ref_values) > 30 and len(curr_values) > 30:
                    # Test de cambio en media (t-test)
                    t_stat, t_p_value = stats.ttest_ind(ref_values, curr_values)
                    
                    # Test de cambio en varianza (Levene's test)
                    levene_stat, levene_p_value = stats.levene(ref_values, curr_values)
                    
                    # Calcular effect sizes
                    mean_ref, mean_curr = ref_values.mean(), curr_values.mean()
                    std_pooled = np.sqrt(((len(ref_values) - 1) * ref_values.var() + 
                                        (len(curr_values) - 1) * curr_values.var()) / 
                                       (len(ref_values) + len(curr_values) - 2))
                    cohens_d = (mean_curr - mean_ref) / std_pooled if std_pooled > 0 else 0
                    
                    results['mean_shift_tests'][col] = {
                        't_statistic': float(t_stat),
                        'p_value': float(t_p_value),
                        'cohens_d': float(cohens_d),
                        'mean_change_pct': float((mean_curr - mean_ref) / mean_ref * 100) if mean_ref != 0 else 0,
                        'drift_detected': t_p_value < 0.05
                    }
                    
                    results['variance_tests'][col] = {
                        'levene_statistic': float(levene_stat),
                        'p_value': float(levene_p_value),
                        'variance_ratio': float(curr_values.var() / ref_values.var()) if ref_values.var() > 0 else 1,
                        'drift_detected': levene_p_value < 0.05
                    }
                    
                    # Marcar cambios significativos
                    if t_p_value < 0.05 or levene_p_value < 0.05:
                        results['significant_changes'].append({
                            'feature': col,
                            'mean_drift': t_p_value < 0.05,
                            'variance_drift': levene_p_value < 0.05,
                            'effect_size': abs(float(cohens_d))
                        })
                        
            except Exception as e:
                logger.warning(f"Error en tests de distribución para {col}: {e}")
                continue
        
        logger.info(f"Tests de distribución completados. Cambios significativos: {len(results['significant_changes'])}")
        return results
    
    def _analyze_feature_importance_drift(self, reference_data: pd.DataFrame, 
                                        current_data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza drift en las features más importantes del modelo."""
        logger.info("Analizando drift en features importantes...")
        
        # Features que típicamente son importantes en recomendación
        important_features = [
            'total_orders', 'unique_products_bought', 'total_transactions',
            'avg_items_per_transaction', 'days_since_last_purchase',
            'unique_customers', 'total_items_sold', 'customer_penetration',
            'interaction_frequency', 'loyalty_score', 'recency_score'
        ]
        
        # Filtrar features que realmente existen
        existing_important_features = [f for f in important_features if f in reference_data.columns]
        
        drift_summary = {
            'important_features_analyzed': existing_important_features,
            'features_with_drift': [],
            'overall_importance_drift_score': 0.0
        }
        
        drift_scores = []
        
        for feature in existing_important_features:
            try:
                ref_values = reference_data[feature].dropna()
                curr_values = current_data[feature].dropna()
                
                if len(ref_values) > 50 and len(curr_values) > 50:
                    # Calcular KS statistic como medida de drift
                    ks_stat, p_value = ks_2samp(ref_values, curr_values)
                    
                    if p_value < 0.05:
                        drift_summary['features_with_drift'].append({
                            'feature': feature,
                            'ks_statistic': float(ks_stat),
                            'p_value': float(p_value),
                            'mean_change_pct': float((curr_values.mean() - ref_values.mean()) / ref_values.mean() * 100) if ref_values.mean() != 0 else 0
                        })
                    
                    drift_scores.append(ks_stat)
                    
            except Exception as e:
                logger.warning(f"Error analizando feature importante {feature}: {e}")
                continue
        
        # Calcular score general de drift en features importantes
        drift_summary['overall_importance_drift_score'] = float(np.mean(drift_scores)) if drift_scores else 0.0
        
        logger.info(f"Análisis de features importantes completado. Features con drift: {len(drift_summary['features_with_drift'])}")
        return drift_summary
    
    def _make_drift_decision(self, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """Toma la decisión final sobre si existe drift significativo."""
        logger.info("Tomando decisión final sobre drift...")
        
        decision = {
            'drift_detected': False,
            'drift_score': 0.0,
            'confidence': 'low',
            'recommendation': 'no_action',
            'significant_tests': [],
            'decision_factors': []
        }
        
        scores = []
        significant_tests = []
        
        # 1. Evaluar tests estadísticos
        if 'statistical_tests' in drift_results:
            significant_drifts = drift_results['statistical_tests']['significant_drifts']
            if significant_drifts:
                scores.append(len(significant_drifts) / 10)  # Normalizar por 10 features
                significant_tests.extend([d['feature'] for d in significant_drifts])
                decision['decision_factors'].append(f"Tests estadísticos: {len(significant_drifts)} features con drift")
        
        # 2. Evaluar PSI
        if 'psi_results' in drift_results:
            psi_drifts = drift_results['psi_results']['features_with_drift']
            if psi_drifts:
                avg_psi = np.mean([d['psi_score'] for d in psi_drifts])
                scores.append(min(avg_psi / 0.25, 1.0))  # Normalizar por umbral alto
                decision['decision_factors'].append(f"PSI: {len(psi_drifts)} features con drift, PSI promedio: {avg_psi:.3f}")
        
        # 3. Evaluar tests de distribución
        if 'distribution_tests' in drift_results:
            dist_changes = drift_results['distribution_tests']['significant_changes']
            if dist_changes:
                scores.append(len(dist_changes) / 10)  # Normalizar
                decision['decision_factors'].append(f"Tests distribución: {len(dist_changes)} features con cambios")
        
        # 4. Evaluar features importantes
        if 'feature_importance_drift' in drift_results:
            importance_drift = drift_results['feature_importance_drift']['overall_importance_drift_score']
            if importance_drift > 0.1:
                scores.append(importance_drift * 2)  # Dar más peso a features importantes
                decision['decision_factors'].append(f"Features importantes: drift score {importance_drift:.3f}")
        
        # Calcular score final
        if scores:
            decision['drift_score'] = float(np.mean(scores))
        else:
            decision['drift_score'] = 0.0
        
        decision['significant_tests'] = list(set(significant_tests))
        
        # Tomar decisión final
        if decision['drift_score'] > self.drift_threshold:
            decision['drift_detected'] = True
            decision['recommendation'] = 'retrain_model'
            
            if decision['drift_score'] > 0.3:
                decision['confidence'] = 'high'
            elif decision['drift_score'] > 0.2:
                decision['confidence'] = 'medium'
            else:
                decision['confidence'] = 'low'
        else:
            decision['drift_detected'] = False
            decision['recommendation'] = 'use_existing_model'
            decision['confidence'] = 'high' if decision['drift_score'] < 0.05 else 'medium'
        
        logger.info(f"Decisión de drift: {decision['drift_detected']} (score: {decision['drift_score']:.3f}, confianza: {decision['confidence']})")
        
        return decision

# Función de utilidad para uso independiente
def detect_sodai_drift(reference_data_path: str, current_data_path: str, 
                      threshold: float = 0.1) -> Dict[str, Any]:
    """
    Función de conveniencia para detectar drift en datos de SodAI.
    
    Args:
        reference_data_path: Ruta a los datos de referencia
        current_data_path: Ruta a los datos actuales
        threshold: Umbral de drift
        
    Returns:
        Diccionario con resultados de detección
    """
    detector = DriftDetector(reference_data_path, current_data_path, threshold)
    return detector.detect_drift()

if __name__ == "__main__":
    # Ejemplo de uso
    import argparse
    
    parser = argparse.ArgumentParser(description='Detector de Drift para SodAI Drinks')
    parser.add_argument('--reference-path', required=True, help='Ruta a datos de referencia')
    parser.add_argument('--current-path', required=True, help='Ruta a datos actuales')
    parser.add_argument('--threshold', type=float, default=0.1, help='Umbral de drift')
    args = parser.parse_args()
    
    result = detect_sodai_drift(args.reference_path, args.current_path, args.threshold)
    print("Detección de drift completada:")
    print(f"- Drift detectado: {result['drift_detected']}")
    print(f"- Score de drift: {result['drift_score']:.3f}")
    print(f"- Recomendación: {result['recommendation']}")
