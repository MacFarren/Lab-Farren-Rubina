import os
import pandas as pd
import joblib
import gradio as gr
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os, json, time
import joblib

def create_folders(**kwargs):
    """
    Crea una carpeta con la fecha de ejecución como nombre y las subcarpetas requeridas.
    
    Utiliza kwargs para obtener la fecha de ejecución mediante el DAG de Airflow.
    La fecha de ejecución se obtiene del contexto de Airflow a través de 'ds' (execution_date en formato YYYY-MM-DD).
    
    Args:
        **kwargs: Contexto de Airflow que incluye 'ds' (execution_date)
    
    Returns:
        str: Ruta de la carpeta principal creada
    """
    # Obtener la fecha de ejecución del contexto de Airflow
    # 'ds' es la fecha de ejecución en formato YYYY-MM-DD
    execution_date = kwargs.get('ds')
    
    if not execution_date:
        # Fallback en caso de que no esté disponible en el contexto
        execution_date = datetime.now().strftime('%Y-%m-%d')
    
    # Crear la carpeta principal con la fecha como nombre
    main_folder = execution_date
    
    # Crear la carpeta principal si no existe
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
        print(f"Carpeta principal creada: {main_folder}")
    else:
        print(f"Carpeta principal ya existe: {main_folder}")
    
    # Definir las subcarpetas requeridas
    subfolders = ['raw', 'splits', 'models']
    
    # Crear cada subcarpeta dentro de la carpeta principal
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            print(f"Subcarpeta creada: {subfolder_path}")
        else:
            print(f"Subcarpeta ya existe: {subfolder_path}")
    
    print(f"Estructura de carpetas completada para la fecha: {execution_date}")
    
    # Retornar la ruta de la carpeta principal para uso posterior
    return os.path.abspath(main_folder)


# Función alternativa usando otros templates de Airflow disponibles
def create_folders_extended(**kwargs):
    """
    Versión extendida que utiliza múltiples templates de Airflow para mayor flexibilidad.
    
    Args:
        **kwargs: Contexto completo de Airflow
    
    Returns:
        dict: Información sobre las carpetas creadas
    """
    # Obtener información del contexto de Airflow
    execution_date = kwargs.get('ds')  # YYYY-MM-DD
    execution_date_nodash = kwargs.get('ds_nodash')  # YYYYMMDD
    dag_id = kwargs.get('dag_id', 'unknown_dag')
    task_id = kwargs.get('task_id', 'unknown_task')
    
    # Usar ds_nodash si está disponible, sino usar ds
    folder_date = execution_date_nodash if execution_date_nodash else execution_date.replace('-', '')
    
    # Crear carpeta con formato: YYYYMMDD
    main_folder = folder_date
    
    # Crear la estructura de carpetas
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
        print(f"[{dag_id}:{task_id}] Carpeta principal creada: {main_folder}")
    
    # Subcarpetas requeridas
    subfolders = ['raw', 'splits', 'models']
    created_folders = []
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            created_folders.append(subfolder_path)
            print(f"[{dag_id}:{task_id}] Subcarpeta creada: {subfolder_path}")
    
    return {
        'main_folder': os.path.abspath(main_folder),
        'execution_date': execution_date,
        'created_subfolders': created_folders,
        'dag_id': dag_id,
        'task_id': task_id
    }


def split_data(**kwargs):
    """
    Lee el archivo data_1.csv de la carpeta raw y aplica un hold out split,
    generando datasets de entrenamiento y prueba. Guarda los nuevos conjuntos
    en la carpeta splits.
    
    Args:
        **kwargs: Contexto de Airflow que incluye 'ds' (execution_date)
        
    Returns:
        dict: Información sobre los datasets creados
    """
    # Obtener la fecha de ejecución del contexto de Airflow
    execution_date = kwargs.get('ds')
    if not execution_date:
        execution_date = datetime.now().strftime('%Y-%m-%d')
    
    # Definir rutas basadas en la estructura de carpetas creada
    main_folder = execution_date
    raw_folder = os.path.join(main_folder, 'raw')
    splits_folder = os.path.join(main_folder, 'splits')
    
    # Ruta del archivo de entrada
    input_file = os.path.join(raw_folder, 'data_1.csv')
    
    # Verificar que el archivo existe
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"El archivo {input_file} no existe. ")
    
    # Leer el dataset
    print(f"Leyendo dataset desde: {input_file}")
    df = pd.read_csv(input_file)
    
    # Mostrar información básica del dataset
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    
    # Detectar la variable objetivo - HiringDecision
    target_column = df.columns[-1]  # Última columna 
    
    # Buscar nombres específicos de variables objetivo para hiring
    common_target_names = ['hiringdecision', 'hiring_decision', 'target', 'label', 'y', 'class', 'outcome', 'hired']
    for col in df.columns:
        # Normalizar tanto el nombre de la columna como los nombres objetivo para comparación
        col_normalized = col.lower().replace(' ', '_').replace('-', '_')
        if col_normalized in common_target_names:
            target_column = col  # Usar el nombre original de la columna
            print(f"Variable objetivo encontrada por nombre: '{col}'")
            break
    
    print(f"Variable objetivo detectada: {target_column}")
    
    # Separar características (X) y variable objetivo (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Mostrar distribución de la variable objetivo
    target_distribution = y.value_counts()
    print(f"Distribución de la variable objetivo:")
    print(target_distribution)
    print(f"Proporciones: {y.value_counts(normalize=True)}")
    
    # Aplicar hold out split con stratify para mantener las proporciones
    # 80% entrenamiento, 20% prueba, semilla fija para reproducibilidad
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    try:
        # Intentar split estratificado para mantener proporciones
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,  # Mantiene la proporción original
            shuffle=True
        )
        print(f"Split estratificado aplicado exitosamente")
        
    except ValueError as e:
        # Si el stratify falla, usar split normal
        print(f"Warning: No se pudo aplicar stratify ({e}). Usando split aleatorio simple.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True
        )
    
    # Recrear los DataFrames completos
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Verificar las proporciones en los conjuntos resultantes
    print(f"\nDataset de entrenamiento: {train_df.shape[0]} filas")
    print(f"Distribución objetivo (entrenamiento): {y_train.value_counts(normalize=True)}")
    
    print(f"\nDataset de prueba: {test_df.shape[0]} filas")
    print(f"Distribución objetivo (prueba): {y_test.value_counts(normalize=True)}")
    
    # Definir rutas de salida
    train_file = os.path.join(splits_folder, 'train_data.csv')
    test_file = os.path.join(splits_folder, 'test_data.csv')
    
    # Guardar los datasets
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\nDatasets guardados:")
    print(f"  - Entrenamiento: {train_file}")
    print(f"  - Prueba: {test_file}")
    
    # Información adicional para logging
    split_info = {
        'original_shape': df.shape,
        'train_shape': train_df.shape,
        'test_shape': test_df.shape,
        'target_column': target_column,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'train_file': os.path.abspath(train_file),
        'test_file': os.path.abspath(test_file),
        'execution_date': execution_date,
        'original_distribution': target_distribution.to_dict(),
        'train_distribution': y_train.value_counts().to_dict(),
        'test_distribution': y_test.value_counts().to_dict()
    }
    
    return split_info


def split_data_with_validation(**kwargs):
    """
    Versión extendida que también crea un conjunto de validación.
    Split: 60% entrenamiento, 20% validación, 20% prueba
    """
    # Similar implementación pero con split adicional para validación
    execution_date = kwargs.get('ds', datetime.now().strftime('%Y-%m-%d'))
    
    main_folder = execution_date
    raw_folder = os.path.join(main_folder, 'raw')
    splits_folder = os.path.join(main_folder, 'splits')
    
    input_file = os.path.join(raw_folder, 'data_1.csv')
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"El archivo {input_file} no existe.")
    
    df = pd.read_csv(input_file)
    target_column = df.columns[-1]
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Primer split: 80% temporal, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Segundo split: del 80% temporal -> 60% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 de 0.8 = 0.2 total
    )
    
    # Crear DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Guardar archivos
    train_df.to_csv(os.path.join(splits_folder, 'train_data.csv'), index=False)
    val_df.to_csv(os.path.join(splits_folder, 'validation_data.csv'), index=False)
    test_df.to_csv(os.path.join(splits_folder, 'test_data.csv'), index=False)
    
    return {
        'train_shape': train_df.shape,
        'validation_shape': val_df.shape,
        'test_shape': test_df.shape
    }


def preprocess_and_train(**kwargs):
    """
    Lee los sets de entrenamiento y prueba, crea un Pipeline con preprocesamiento
    usando ColumnTransformers, entrena un modelo RandomForest y guarda el pipeline.
    
    El preprocesamiento incluye:
    - Imputación de valores faltantes
    - Escalado para variables numéricas
    - One-Hot Encoding para variables categóricas
    - Manejo robusto de diferentes tipos de datos
    
    Args:
        **kwargs: Contexto de Airflow que incluye 'ds' (execution_date)
        
    Returns:
        dict: Métricas y información del modelo entrenado
    """
    # Obtener la fecha de ejecución del contexto de Airflow
    execution_date = kwargs.get('ds')
    if not execution_date:
        execution_date = datetime.now().strftime('%Y-%m-%d')
    
    # Definir rutas basadas en la estructura de carpetas
    main_folder = execution_date
    splits_folder = os.path.join(main_folder, 'splits')
    models_folder = os.path.join(main_folder, 'models')
    
    # Rutas de los archivos de datos
    train_file = os.path.join(splits_folder, 'train_data.csv')
    test_file = os.path.join(splits_folder, 'test_data.csv')
    
    # Verificar que los archivos existen
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Archivo de entrenamiento no encontrado: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Archivo de prueba no encontrado: {test_file}")
    
    # Cargar los datasets
    print(f"Cargando datasets desde {splits_folder}")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Dataset de entrenamiento: {train_df.shape}")
    print(f"Dataset de prueba: {test_df.shape}")
    
    # Detectar la variable objetivo - priorizar HiringDecision
    target_column = train_df.columns[-1]  # Última columna por defecto
    
    # Verificar nombres específicos de variable objetivo para hiring
    common_target_names = ['hiringdecision', 'hiring_decision', 'hired', 'target', 'label', 'y', 'class', 'outcome', 'decision']
    for col in train_df.columns:
        # Normalizar tanto el nombre de la columna como los nombres objetivo para comparación
        col_normalized = col.lower().replace(' ', '_').replace('-', '_')
        if col_normalized in common_target_names:
            target_column = col  # Usar el nombre original de la columna
            print(f"Variable objetivo encontrada por nombre: '{col}'")
            break
    
    print(f"Variable objetivo: {target_column}")
    
    # Separar características y variable objetivo
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    # Análisis de tipos de datos para el preprocesamiento
    print(f"\nAnálisis de características:")
    print(f"Total de características: {X_train.shape[1]}")
    
    # Identificar columnas numéricas y categóricas
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # También verificar columnas que parecen categóricas pero son numéricas (ej: 0,1 binarias)
    potential_categorical = []
    for col in numeric_features.copy():
        unique_vals = X_train[col].nunique()
        if unique_vals <= 10:  # Pocas categorías únicas
            print(f"  - {col}: {unique_vals} valores únicos - tratada como categórica")
            potential_categorical.append(col)
            categorical_features.append(col)
            numeric_features.remove(col)
    
    print(f"Características numéricas: {len(numeric_features)} -> {numeric_features}")
    print(f"Características categóricas: {len(categorical_features)} -> {categorical_features}")
    
    # Verificar valores faltantes
    missing_info = X_train.isnull().sum()
    print(f"\nValores faltantes por columna:")
    for col, missing_count in missing_info.items():
        if missing_count > 0:
            pct = (missing_count / len(X_train)) * 100
            print(f"  - {col}: {missing_count} ({pct:.1f}%)")
    
    # Crear transformadores para preprocesamiento
    
    # Transformador para características numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Imputación con mediana (robusto a outliers)
        ('scaler', StandardScaler())  # Escalado estándar
    ])
    
    # Transformador para características categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputación con moda
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))  # One-hot encoding
    ])
    
    # Combinar transformadores usando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough',  # Mantener columnas no especificadas
        verbose_feature_names_out=False
    )
    
    # Configurar RandomForest con parámetros balanceados
    rf_classifier = RandomForestClassifier(
        n_estimators=100,           # Número de árboles
        max_depth=10,              # Profundidad máxima para evitar overfitting
        min_samples_split=5,       # Mínimo de muestras para dividir
        min_samples_leaf=2,        # Mínimo de muestras en hojas
        class_weight='balanced',   # Balancear clases automáticamente
        random_state=42,          # Semilla para reproducibilidad
        n_jobs=-1                 # Usar todos los cores disponibles
    )
    
    # Crear el pipeline completo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf_classifier)
    ])
    
    print(f"\nEntrenando modelo RandomForest...")
    print(f"Configuración del modelo:")
    print(f"  - Estimadores: {rf_classifier.n_estimators}")
    print(f"  - Max profundidad: {rf_classifier.max_depth}")
    print(f"  - Balanceo de clases: {rf_classifier.class_weight}")
    
    # Entrenar el pipeline
    pipeline.fit(X_train, y_train)
    
    print(f"Entrenamiento completado!")
    
    # Hacer predicciones en el conjunto de prueba
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    # Para F1-score de la clase positiva, identificar cuál es la clase positiva
    unique_classes = sorted(y_train.unique())
    print(f"Clases únicas en '{target_column}': {unique_classes}")
    
    # Para HiringDecision, la clase positiva es típicamente la que indica contratación exitosa
    if len(unique_classes) == 2:
        # Clasificación binaria
        positive_class = max(unique_classes)  # Por defecto, la clase con valor mayor
        
        # Buscar específicamente valores que indican contratación positiva
        positive_variations = [
            'hired', 'yes', 'true', '1', 'positive', 'accepted', 'approved', 
            'selected', 'offer_made', 'successful', 'pass'
        ]
        negative_variations = [
            'not_hired', 'no', 'false', '0', 'negative', 'rejected', 'declined',
            'not_selected', 'unsuccessful', 'fail'
        ]
        
        # Verificar las clases para identificar cuál es positiva
        for cls in unique_classes:
            cls_str = str(cls).lower().replace(' ', '_').replace('-', '_')
            if cls_str in positive_variations:
                positive_class = cls
                print(f"Clase positiva detectada: '{cls}' (indica contratación exitosa)")
                break
            elif cls_str in negative_variations:
                # Si encontramos la negativa, la otra debe ser positiva
                positive_class = [c for c in unique_classes if c != cls][0]
                print(f"Clase positiva inferida: '{positive_class}' (contraparte de '{cls}')")
                break
        else:
            print(f"Usando clase con valor mayor como positiva: '{positive_class}'")
            
    else:
        # Multi-clase: usar 'macro' average
        positive_class = 'macro'
        print(f"Clasificación multi-clase detectada, usando F1-score macro")
    
    # Calcular F1-score
    if len(unique_classes) == 2:
        f1 = f1_score(y_test, y_pred, pos_label=positive_class)
        print(f"\nMÉTRICAS DEL MODELO:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (clase positiva '{positive_class}'): {f1:.4f}")
    else:
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"\nMÉTRICAS DEL MODELO:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (macro): {f1:.4f}")
    
    # Mostrar reporte detallado
    print(f"\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Guardar el pipeline entrenado usando joblib
    model_filename = 'trained_pipeline.joblib'
    model_path = os.path.join(models_folder, model_filename)
    
    joblib.dump(pipeline, model_path)
    print(f"Pipeline guardado en: {model_path}")
    
    # Información sobre las características más importantes (Feature Importance)
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        # Obtener nombres de características después del preprocesamiento
        try:
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            importances = pipeline.named_steps['classifier'].feature_importances_
            
            # Crear DataFrame con importancias
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 características más importantes:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
                
        except Exception as e:
            print(f"No se pudieron obtener las importancias de características: {e}")
    
    # Preparar información de retorno
    model_info = {
        'model_path': os.path.abspath(model_path),
        'execution_date': execution_date,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'target_column': target_column,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'unique_classes': unique_classes,
        'positive_class': str(positive_class) if positive_class != 'macro' else 'macro',
        'model_params': {
            'n_estimators': rf_classifier.n_estimators,
            'max_depth': rf_classifier.max_depth,
            'class_weight': str(rf_classifier.class_weight)
        }
    }
    
    return model_info

def gradio_interface(**kwargs):
    """
    Interfaz Gradio que permite subir un JSON y obtener predicción.
    Lanza Gradio en background y termina la tarea rápidamente.
    """
    import threading
    import time
    
    # 1) Fecha de ejecución (Airflow) -> carpeta del run
    execution_date = kwargs.get('ds') or datetime.now().strftime('%Y-%m-%d')

    # 2) Rutas (usar directorio actual como base - compatible con Docker)
    current_dir = os.getcwd()
    print(f"[Gradio] Directorio actual: {current_dir}")
    main_folder = os.path.join(current_dir, execution_date)
    models_folder = os.path.join(main_folder, 'models')
    model_path = os.path.join(models_folder, 'trained_pipeline.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    print(f"[Gradio] Cargando modelo desde: {model_path}")
    pipeline = joblib.load(model_path)

    def procesar_archivo_json(archivo):
        try:
            if archivo is None:
                return "Error: No se ha subido ningún archivo."
            with open(archivo.name, 'r', encoding='utf-8') as f:
                datos = json.load(f)

            # Soporta dict (1 fila) o lista de dicts (N filas)
            X = pd.DataFrame([datos]) if isinstance(datos, dict) else pd.DataFrame(datos)

            yhat = pipeline.predict(X)
            # Probabilidades
            proba_txt = ""
            try:
                probs = pipeline.predict_proba(X)
                clf = getattr(pipeline, "named_steps", {}).get("classifier", None)
                classes_ = getattr(clf, "classes_", None) if clf is not None else None
                if classes_ is None:
                    classes_ = getattr(pipeline, "classes_", None)
                if classes_ is not None:
                    proba_pairs = [(str(c), float(p)) for c, p in zip(classes_, probs[0])]
                    proba_txt = "\nProbabilidades:\n" + "\n".join(
                        f"- {c}: {p:.2%}" for c, p in proba_pairs
                    )
            except Exception:
                proba_txt = "\nProbabilidades no disponibles."

            decision = yhat[0]
            cabecera = f"RESULTADO DE LA PREDICCIÓN\n\nDecisión: {'CONTRATAR' if decision == 1 else 'NO CONTRATAR'}"
            return cabecera + proba_txt

        except json.JSONDecodeError:
            return "Error: El archivo no tiene un formato JSON válido."
        except Exception as e:
            return f"Error al procesar el archivo: {e}"

    def launch_gradio():
        """Función para lanzar Gradio en un hilo separado"""
        interfaz = gr.Interface(
            fn=procesar_archivo_json,
            inputs=gr.File(label="Subir archivo JSON con datos del candidato", file_types=[".json"]),
            outputs=gr.Textbox(label="Resultado de la Predicción", lines=10),
            title="Predicción de Contratación - Subida de Archivo",
            description="Sube un archivo JSON con los datos del candidato para obtener una predicción."
        )
        
        print("[Gradio] Lanzando interfaz en hilo separado...")
        url = interfaz.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            prevent_thread_lock=True,
            quiet=True
        )
        
        # Esperar un momento para obtener la URL
        time.sleep(5)
        
        try:
            share_url = getattr(interfaz, 'share_url', 'URL no disponible aún')
            print(f"[Gradio]  URL PÚBLICA: {share_url}")
            print(f"[Gradio]  URL LOCAL: http://localhost:7860")
        except Exception as e:
            print(f"[Gradio] URL no disponible: {e}")

    # Lanzar Gradio en un hilo daemon (se cierra cuando termina el programa)
    gradio_thread = threading.Thread(target=launch_gradio, daemon=True)
    gradio_thread.start()
    
    # Esperar un poco para que se inicie
    time.sleep(8)
    
    print("[Gradio]  Servidor iniciado en background - Tarea completada")
    print("[Gradio]  La interfaz seguirá corriendo independientemente")
    print("[Gradio]  Revisar los logs para obtener la URL pública")

    return {
        'interface_url': 'http://localhost:7860',
        'model_path': os.path.abspath(model_path),
        'execution_date': execution_date,
        'status': 'launched_in_background'
    }


def smoke_test_gradio(**kwargs):
    """
    Función alternativa: solo prueba que el modelo funcione sin lanzar Gradio.
    Útil para debugging y testing rápido.
    """
    execution_date = kwargs.get('ds') or datetime.now().strftime('%Y-%m-%d')
    
    main_folder = (
        os.path.join("/opt/airflow", execution_date)
        if os.path.exists("/opt/airflow") else execution_date
    )
    models_folder = os.path.join(main_folder, 'models')
    model_path = os.path.join(models_folder, 'trained_pipeline.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    print(f"[SmokeTest]  Modelo encontrado: {model_path}")
    
    # Cargar modelo
    pipeline = joblib.load(model_path)
    print(f"[SmokeTest]  Modelo cargado exitosamente")
    
    # Datos de prueba (vale_data.json)
    test_data = {
        "Age": 25,
        "Gender": 1,
        "EducationLevel": 3,
        "ExperienceYears": 2,
        "PreviousCompanies": 2,
        "DistanceFromCompany": 40,
        "InterviewScore": 98.0,
        "SkillScore": 97.0,
        "PersonalityScore": 99.0,
        "RecruitmentStrategy": 2
    }
    
    # Hacer predicción de prueba
    X_test = pd.DataFrame([test_data])
    prediction = pipeline.predict(X_test)[0]
    probabilities = pipeline.predict_proba(X_test)[0]
    
    print(f"[SmokeTest]  Predicción de prueba:")
    print(f"[SmokeTest]   - Decisión: {'CONTRATAR' if prediction == 1 else 'NO CONTRATAR'}")
    print(f"[SmokeTest]   - Probabilidades: {probabilities}")
    print(f"[SmokeTest]  EL MODELO FUNCIONA CORRECTAMENTE")
    
    print(f"[SmokeTest]  Para usar Gradio manualmente:")
    print(f"[SmokeTest]   1. Ejecutar: docker exec -it lab9-airflow-1 python")
    print(f"[SmokeTest]   2. Importar y ejecutar gradio_interface()")
    
    return {
        'model_path': os.path.abspath(model_path),
        'test_prediction': int(prediction),
        'test_probabilities': probabilities.tolist(),
        'status': 'smoke_test_completed'
    }


def load_and_predict_single(input_data, **kwargs):
    """
    Función auxiliar para realizar una predicción individual sin interfaz web.
    
    Args:
        input_data (dict): Diccionario con los datos del candidato
        **kwargs: Contexto de Airflow
        
    Returns:
        dict: Resultado de la predicción con probabilidades
    """
    # Obtener la fecha de ejecución
    execution_date = kwargs.get('ds', datetime.now().strftime('%Y-%m-%d'))
    
    # Cargar el modelo
    main_folder = execution_date
    models_folder = os.path.join(main_folder, 'models')
    model_path = os.path.join(models_folder, 'trained_pipeline.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    pipeline = joblib.load(model_path)
    
    # Convertir a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Realizar predicción
    prediction = pipeline.predict(input_df)[0]
    probabilities = pipeline.predict_proba(input_df)[0]
    classes = pipeline.named_steps['classifier'].classes_
    
    return {
        'prediction': int(prediction),
        'prediction_label': 'Contratar' if prediction == 1 else 'No Contratar',
        'probabilities': {
            str(cls): float(prob) for cls, prob in zip(classes, probabilities)
        },
        'confidence': float(max(probabilities)),
        'model_path': model_path
    }



def smoke_test_gradio(**context):
    """
    Verifica que el modelo entrenado y los datos de prueba estén disponibles 
    y funcionen correctamente antes de desplegar Gradio.

    Comprueba:
      • Que exista `best_model.joblib` en `/opt/airflow/<ds>/models/`.
      • Que exista `X_test.csv` en `/opt/airflow/<ds>/splits/`.
      • Que el modelo cargue y prediga sin errores sobre unas pocas filas.

    Lanza FileNotFoundError si falta algún archivo.
    Imprime 'Smoke test OK' si todo funciona.
    """

    import os, joblib, pandas as pd
    ds = context["ds"]
    base_dir = os.path.join("/opt/airflow", ds)  
    model_path = os.path.join(base_dir, "models", "best_model.joblib")
    xtest_path = os.path.join(base_dir, "splits", "X_test.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    if not os.path.exists(xtest_path):
        raise FileNotFoundError(f"No existe X_test: {xtest_path}")

    pipe = joblib.load(model_path)
    X_test = pd.read_csv(xtest_path)

    # toma 5 filas (o 1 si el test es chico)
    n = min(5, len(X_test))
    _ = pipe.predict(X_test.iloc[:n])
    print(f"Smoke test OK: modelo cargó y predijo sobre {n} filas.")