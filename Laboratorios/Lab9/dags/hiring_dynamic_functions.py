# Archivo: hiring_dynamic_functions.py
# Funciones para el pipeline dinámico (Sección 2) - Pipeline paralelizado

import os
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from glob import glob


def create_folders(**kwargs):
    """
    Crea una carpeta con la fecha de ejecución como nombre y las subcarpetas requeridas
    para el pipeline dinámico (Sección 2).
    
    Subcarpetas creadas:
    - raw: Para datos originales descargados
    - preprocessed: Para datos combinados y preprocesados
    - splits: Para conjuntos de entrenamiento y prueba
    - models: Para modelos entrenados
    
    Args:
        **kwargs: Contexto de Airflow que incluye 'ds' (execution_date)
    
    Returns:
        str: Ruta de la carpeta principal creada
    """
    # Obtener la fecha de ejecución del contexto de Airflow
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
    
    # Definir las subcarpetas requeridas para el pipeline dinámico
    subfolders = ['raw', 'preprocessed', 'splits', 'models']
    
    # Crear cada subcarpeta dentro de la carpeta principal
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            print(f"Subcarpeta creada: {subfolder_path}")
        else:
            print(f"Subcarpeta ya existe: {subfolder_path}")
    
    print(f"Estructura de carpetas para pipeline dinámico completada para la fecha: {execution_date}")
    
    # Retornar la ruta de la carpeta principal para uso posterior
    return os.path.abspath(main_folder)


def load_and_merge(**kwargs):
    """
    Lee los archivos data_1.csv y data_2.csv (si está disponible) desde la carpeta raw,
    los concatena y genera un archivo resultante guardándolo en la carpeta preprocessed.
    
    Args:
        **kwargs: Contexto de Airflow que incluye 'ds' (execution_date)
        
    Returns:
        dict: Información sobre los archivos procesados y el resultado
    """
    # Obtener la fecha de ejecución del contexto de Airflow
    execution_date = kwargs.get('ds')
    if not execution_date:
        execution_date = datetime.now().strftime('%Y-%m-%d')
    
    # Definir rutas basadas en la estructura de carpetas creada
    main_folder = execution_date
    raw_folder = os.path.join(main_folder, 'raw')
    preprocessed_folder = os.path.join(main_folder, 'preprocessed')
    
    # Rutas de los archivos de entrada
    data1_file = os.path.join(raw_folder, 'data_1.csv')
    data2_file = os.path.join(raw_folder, 'data_2.csv')
    
    # Lista para almacenar los DataFrames encontrados
    dataframes = []
    files_info = []
    
    # Verificar y cargar data_1.csv
    if os.path.exists(data1_file):
        print(f"Cargando data_1.csv desde: {data1_file}")
        df1 = pd.read_csv(data1_file)
        dataframes.append(df1)
        files_info.append({
            'file': 'data_1.csv',
            'path': data1_file,
            'shape': df1.shape,
            'columns': list(df1.columns)
        })
        print(f"  - data_1.csv: {df1.shape[0]} filas, {df1.shape[1]} columnas")
    else:
        print(f"Advertencia: data_1.csv no encontrado en {data1_file}")
    
    # Verificar y cargar data_2.csv (si está disponible)
    if os.path.exists(data2_file):
        print(f"Cargando data_2.csv desde: {data2_file}")
        df2 = pd.read_csv(data2_file)
        dataframes.append(df2)
        files_info.append({
            'file': 'data_2.csv',
            'path': data2_file,
            'shape': df2.shape,
            'columns': list(df2.columns)
        })
        print(f"  - data_2.csv: {df2.shape[0]} filas, {df2.shape[1]} columnas")
    else:
        print(f"data_2.csv no encontrado en {data2_file} (esto es esperado para fechas antes de nov 2024)")
    
    # Verificar que se encontró al menos un archivo
    if not dataframes:
        raise FileNotFoundError(f"No se encontraron archivos de datos en {raw_folder}")
    
    print(f"Archivos encontrados: {len(dataframes)}")
    
    # Concatenar los DataFrames disponibles
    if len(dataframes) == 1:
        print("Solo un archivo disponible, copiando sin concatenar")
        merged_df = dataframes[0].copy()
    else:
        print("Concatenando múltiples archivos...")
        
        # Verificar que las columnas sean compatibles
        columns_sets = [set(df.columns) for df in dataframes]
        if not all(cols == columns_sets[0] for cols in columns_sets):
            print("Advertencia: Las columnas de los archivos no coinciden exactamente")
            for i, (df, info) in enumerate(zip(dataframes, files_info)):
                print(f"  {info['file']}: {info['columns']}")
            
            # Usar intersección de columnas o manejar diferencias
            common_columns = set.intersection(*columns_sets)
            if common_columns:
                print(f"Usando columnas comunes: {sorted(common_columns)}")
                dataframes = [df[sorted(common_columns)] for df in dataframes]
            else:
                raise ValueError("No hay columnas comunes entre los archivos")
        
        # Concatenar los DataFrames
        merged_df = pd.concat(dataframes, axis=0, ignore_index=True)
        print(f"Concatenación completada: {merged_df.shape[0]} filas totales")
    
    # Información sobre el dataset combinado
    print(f"\nDataset combinado:")
    print(f"  - Dimensiones: {merged_df.shape}")
    print(f"  - Columnas: {list(merged_df.columns)}")
    
    # Verificar valores faltantes
    missing_info = merged_df.isnull().sum()
    total_missing = missing_info.sum()
    if total_missing > 0:
        print(f"  - Valores faltantes encontrados: {total_missing}")
        for col, missing_count in missing_info.items():
            if missing_count > 0:
                pct = (missing_count / len(merged_df)) * 100
                print(f"    * {col}: {missing_count} ({pct:.1f}%)")
    else:
        print("  - Sin valores faltantes")
    
    # Verificar duplicados
    duplicates = merged_df.duplicated().sum()
    if duplicates > 0:
        print(f"  - Filas duplicadas encontradas: {duplicates}")
        # Opcionalmente, remover duplicados
        merged_df = merged_df.drop_duplicates(ignore_index=True)
        print(f"  - Después de remover duplicados: {merged_df.shape}")
    
    # Guardar el dataset combinado
    output_file = os.path.join(preprocessed_folder, 'merged_data.csv')
    merged_df.to_csv(output_file, index=False)
    
    print(f"\nDataset combinado guardado en: {output_file}")
    
    # Preparar información de retorno
    merge_info = {
        'execution_date': execution_date,
        'files_processed': files_info,
        'output_file': os.path.abspath(output_file),
        'final_shape': merged_df.shape,
        'columns': list(merged_df.columns),
        'missing_values': missing_info.to_dict(),
        'duplicates_removed': duplicates,
        'files_count': len(dataframes)
    }
    
    return merge_info


def split_data(**kwargs):
    """
    Lee la data guardada en la carpeta preprocessed y realiza un hold out sobre esta data.
    Crea un conjunto de entrenamiento y uno de prueba. Mantiene una semilla y 20% para 
    el conjunto de prueba. Guarda los conjuntos resultantes en la carpeta splits.
    
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
    preprocessed_folder = os.path.join(main_folder, 'preprocessed')
    splits_folder = os.path.join(main_folder, 'splits')
    
    # Ruta del archivo de entrada (datos combinados)
    input_file = os.path.join(preprocessed_folder, 'merged_data.csv')
    
    # Verificar que el archivo existe
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"El archivo {input_file} no existe. Asegúrese de que la función load_and_merge haya ejecutado correctamente.")
    
    # Leer el dataset
    print(f"Leyendo dataset preprocessed desde: {input_file}")
    df = pd.read_csv(input_file)
    
    # Mostrar información básica del dataset
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    
    # Detectar la variable objetivo - priorizar HiringDecision
    target_column = df.columns[-1]  # Última columna por defecto
    
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
        # Si el stratify falla (ej: muy pocas muestras por clase), usar split normal
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
        'test_distribution': y_test.value_counts().to_dict(),
        'input_file': os.path.abspath(input_file)
    }
    
    return split_info


def train_model(model, **kwargs):
    """
    Recibe un modelo de clasificación, lee el conjunto de entrenamiento desde la carpeta splits,
    crea y aplica un Pipeline con una etapa de preprocesamiento, añade una etapa de entrenamiento
    utilizando el modelo ingresado, y guarda el pipeline entrenado con un nombre identificable.
    
    Args:
        model: Instancia del modelo de clasificación a entrenar
        **kwargs: Contexto de Airflow que incluye 'ds' (execution_date)
        
    Returns:
        dict: Información sobre el modelo entrenado y su desempeño
    """
    # Obtener la fecha de ejecución del contexto de Airflow
    execution_date = kwargs.get('ds')
    if not execution_date:
        execution_date = datetime.now().strftime('%Y-%m-%d')
    
    # Definir rutas basadas en la estructura de carpetas
    main_folder = execution_date
    splits_folder = os.path.join(main_folder, 'splits')
    models_folder = os.path.join(main_folder, 'models')
    
    # Ruta del archivo de entrenamiento
    train_file = os.path.join(splits_folder, 'train_data.csv')
    
    # Verificar que el archivo existe
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Archivo de entrenamiento no encontrado: {train_file}")
    
    # Cargar el dataset de entrenamiento
    print(f"Cargando dataset de entrenamiento desde: {train_file}")
    train_df = pd.read_csv(train_file)
    
    print(f"Dataset de entrenamiento: {train_df.shape}")
    
    # Detectar la variable objetivo
    target_column = train_df.columns[-1]  # Última columna por defecto
    
    # Verificar nombres específicos de variable objetivo para hiring
    common_target_names = ['hiringdecision', 'hiring_decision', 'hired', 'target', 'label', 'y', 'class', 'outcome', 'decision']
    for col in train_df.columns:
        col_normalized = col.lower().replace(' ', '_').replace('-', '_')
        if col_normalized in common_target_names:
            target_column = col
            break
    
    print(f"Variable objetivo: {target_column}")
    
    # Separar características y variable objetivo
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    
    # Identificar el tipo de modelo para el nombre del archivo
    model_name = type(model).__name__
    
    print(f"Entrenando modelo: {model_name}")
    print(f"Características: {X_train.shape[1]}")
    print(f"Muestras de entrenamiento: {X_train.shape[0]}")
    
    # Análisis de tipos de datos para el preprocesamiento
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # También verificar columnas que parecen categóricas pero son numéricas
    for col in numeric_features.copy():
        unique_vals = X_train[col].nunique()
        if unique_vals <= 10:  # Pocas categorías únicas
            categorical_features.append(col)
            numeric_features.remove(col)
    
    print(f"Características numéricas: {len(numeric_features)} -> {numeric_features}")
    print(f"Características categóricas: {len(categorical_features)} -> {categorical_features}")
    
    # Crear transformadores para preprocesamiento
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Combinar transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    # Crear el pipeline completo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    print(f"Entrenando pipeline con {model_name}...")
    
    # Entrenar el pipeline
    pipeline.fit(X_train, y_train)
    
    print(f"Entrenamiento de {model_name} completado!")
    
    # Crear nombre descriptivo para el archivo
    model_filename = f'pipeline_{model_name.lower()}_{execution_date}.joblib'
    model_path = os.path.join(models_folder, model_filename)
    
    # Guardar el pipeline entrenado
    joblib.dump(pipeline, model_path)
    print(f"Pipeline {model_name} guardado en: {model_path}")
    
    # Preparar información de retorno
    model_info = {
        'model_name': model_name,
        'model_path': os.path.abspath(model_path),
        'model_filename': model_filename,
        'execution_date': execution_date,
        'train_shape': X_train.shape,
        'target_column': target_column,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'train_file': os.path.abspath(train_file)
    }
    
    return model_info


def evaluate_models(**kwargs):
    """
    Evalúa todos los modelos entrenados desde la carpeta models, evalúa su desempeño
    mediante accuracy en el conjunto de prueba y selecciona el mejor modelo obtenido.
    Guarda el mejor modelo como archivo .joblib e imprime el nombre del modelo seleccionado
    y el accuracy obtenido.
    
    Args:
        **kwargs: Contexto de Airflow que incluye 'ds' (execution_date)
        
    Returns:
        dict: Información sobre la evaluación y el mejor modelo
    """
    # Obtener la fecha de ejecución del contexto de Airflow
    execution_date = kwargs.get('ds')
    if not execution_date:
        execution_date = datetime.now().strftime('%Y-%m-%d')
    
    # Definir rutas basadas en la estructura de carpetas
    main_folder = execution_date
    splits_folder = os.path.join(main_folder, 'splits')
    models_folder = os.path.join(main_folder, 'models')
    
    # Rutas de archivos
    test_file = os.path.join(splits_folder, 'test_data.csv')
    
    # Verificar que el archivo de prueba existe
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Archivo de prueba no encontrado: {test_file}")
    
    # Cargar el dataset de prueba
    print(f"Cargando dataset de prueba desde: {test_file}")
    test_df = pd.read_csv(test_file)
    
    print(f"Dataset de prueba: {test_df.shape}")
    
    # Detectar la variable objetivo
    target_column = test_df.columns[-1]
    
    common_target_names = ['hiringdecision', 'hiring_decision', 'hired', 'target', 'label', 'y', 'class', 'outcome', 'decision']
    for col in test_df.columns:
        col_normalized = col.lower().replace(' ', '_').replace('-', '_')
        if col_normalized in common_target_names:
            target_column = col
            break
    
    print(f"Variable objetivo: {target_column}")
    
    # Separar características y variable objetivo
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    # Encontrar todos los modelos entrenados en la carpeta models
    model_pattern = os.path.join(models_folder, 'pipeline_*.joblib')
    model_files = glob(model_pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No se encontraron modelos entrenados en {models_folder}")
    
    print(f"Modelos encontrados para evaluación: {len(model_files)}")
    for model_file in model_files:
        print(f"  - {os.path.basename(model_file)}")
    
    # Evaluar cada modelo
    model_results = []
    
    for model_file in model_files:
        try:
            # Extraer nombre del modelo del nombre del archivo
            model_filename = os.path.basename(model_file)
            model_name = model_filename.replace('pipeline_', '').replace(f'_{execution_date}.joblib', '').title()
            
            print(f"\nEvaluando {model_name}...")
            
            # Cargar el pipeline
            pipeline = joblib.load(model_file)
            
            # Realizar predicciones
            y_pred = pipeline.predict(X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calcular F1-score
            unique_classes = sorted(y_test.unique())
            if len(unique_classes) == 2:
                positive_class = max(unique_classes)
                f1 = f1_score(y_test, y_pred, pos_label=positive_class)
            else:
                f1 = f1_score(y_test, y_pred, average='macro')
            
            print(f"  - Accuracy: {accuracy:.4f}")
            print(f"  - F1-Score: {f1:.4f}")
            
            # Guardar resultados
            model_results.append({
                'model_name': model_name,
                'model_file': model_file,
                'model_filename': model_filename,
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': y_pred
            })
            
        except Exception as e:
            print(f"Error evaluando {model_file}: {e}")
            continue
    
    if not model_results:
        raise RuntimeError("No se pudieron evaluar ninguno de los modelos")
    
    # Encontrar el mejor modelo basado en accuracy
    best_model = max(model_results, key=lambda x: x['accuracy'])
    
    print(f"\n" + "="*50)
    print(f"RESULTADOS DE LA EVALUACIÓN:")
    print(f"="*50)
    
    # Mostrar resultados de todos los modelos ordenados por accuracy
    sorted_results = sorted(model_results, key=lambda x: x['accuracy'], reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        status = "🏆 MEJOR" if result == best_model else f"#{i}"
        print(f"{status} - {result['model_name']}")
        print(f"    Accuracy: {result['accuracy']:.4f}")
        print(f"    F1-Score: {result['f1_score']:.4f}")
    
    print(f"="*50)
    print(f"MODELO SELECCIONADO: {best_model['model_name']}")
    print(f"ACCURACY OBTENIDO: {best_model['accuracy']:.4f}")
    print(f"="*50)
    
    # Guardar el mejor modelo con un nombre especial
    best_model_filename = 'best_model.joblib'
    best_model_path = os.path.join(models_folder, best_model_filename)
    
    # Copiar el mejor modelo
    import shutil
    shutil.copy2(best_model['model_file'], best_model_path)
    print(f"Mejor modelo guardado como: {best_model_path}")
    
    # Mostrar reporte detallado del mejor modelo
    print(f"\nReporte detallado del mejor modelo ({best_model['model_name']}):")
    print(classification_report(y_test, best_model['predictions']))
    
    # Preparar información de retorno
    evaluation_info = {
        'execution_date': execution_date,
        'models_evaluated': len(model_results),
        'best_model_name': best_model['model_name'],
        'best_model_accuracy': float(best_model['accuracy']),
        'best_model_f1': float(best_model['f1_score']),
        'best_model_path': os.path.abspath(best_model_path),
        'best_model_original_file': best_model['model_file'],
        'all_results': [
            {
                'model_name': r['model_name'],
                'accuracy': float(r['accuracy']),
                'f1_score': float(r['f1_score'])
            }
            for r in sorted_results
        ],
        'test_file': os.path.abspath(test_file),
        'test_shape': X_test.shape,
        'target_column': target_column
    }
    
    return evaluation_info