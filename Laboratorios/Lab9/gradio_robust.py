#!/usr/bin/env python3
"""
Gradio interface robusto para predicción de contratación
"""
import gradio as gr
import joblib
import json
import pandas as pd
import os
import traceback

# Configuración
os.chdir('/opt/airflow/OneDrive/Documentos/LabMDS/Lab-Farren-Rubina/Lab9')

def load_model():
    """Cargar el mejor modelo disponible"""
    model_paths = [
        '2025-11-04/models/best_model.joblib',
        '2025-11-04/models/trained_pipeline.joblib',
        '2025-11-04/models/pipeline_randomforestclassifier_2025-11-04.joblib'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Cargando modelo desde: {path}")
            return joblib.load(path)
    
    raise FileNotFoundError("No se encontró ningún modelo entrenado")

# Cargar modelo
try:
    model = joblib.load('2025-11-04/models/trained_pipeline.joblib')
    print("Modelo cargado correctamente!")
except:
    model = joblib.load('2025-11-04/models/best_model.joblib')
    print("Modelo alternativo cargado!")

def predict_hiring(file):
    """Función de predicción mejorada para Gradio"""
    if file is None:
        return " Por favor, sube un archivo JSON con los datos del candidato."
    
    try:
        # Leer el archivo JSON
        print(f"Procesando archivo: {type(file)} - {file}")
        
        # En Gradio 3.34, file es un objeto TemporaryFile
        if hasattr(file, 'name'):
            file_path = file.name
        else:
            file_path = file
            
        print(f"Ruta del archivo: {file_path}")
        
        # Leer contenido del archivo
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            return " El archivo está vacío"
        
        print(f"Contenido leído: {content[:100]}...")
        
        # Parsear JSON
        data = json.loads(content)
        
        if isinstance(data, list) and len(data) > 0:
            candidate = data[0]
        elif isinstance(data, dict):
            candidate = data
        else:
            return " Formato JSON inválido. Se esperaba un objeto o lista de objetos."
        
        print(f"Datos del candidato: {candidate}")
        
        # Crear DataFrame
        df = pd.DataFrame([candidate])
        
        # Mapeo de variables categóricas
        mappings = {
            'Gender': {0: 'Female', 1: 'Male'},
            'EducationLevel': {1: 'High School', 2: 'Bachelor', 3: 'Master', 4: 'PhD'},
            'RecruitmentStrategy': {1: 'Agency', 2: 'Direct', 3: 'University'}
        }
        
        # Aplicar mappings
        for col, mapping in mappings.items():
            if col in df.columns:
                original_values = df[col].values
                df[col] = df[col].map(mapping).fillna(df[col])
                print(f"Mapeado {col}: {original_values} -> {df[col].values}")
        
        print(f"DataFrame procesado:\n{df}")
        
        # Hacer predicción
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        # Formatear resultado
        decision = " CONTRATAR" if prediction == 1 else " NO CONTRATAR"
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        result = f"""
 **DECISIÓN DE CONTRATACIÓN**: {decision}
 **Confianza**: {confidence*100:.1f}%

 **Probabilidades detalladas**:
• No Contratar: {probabilities[0]*100:.1f}%
• Contratar: {probabilities[1]*100:.1f}%

 **Perfil del candidato**:
• Edad: {candidate.get('Age', 'N/A')} años
• Género: {df['Gender'].iloc[0] if 'Gender' in df.columns else 'N/A'}
• Educación: {df['EducationLevel'].iloc[0] if 'EducationLevel' in df.columns else 'N/A'}
• Experiencia: {candidate.get('ExperienceYears', 'N/A')} años
• Empresas anteriores: {candidate.get('PreviousCompanies', 'N/A')}
• Distancia: {candidate.get('DistanceFromCompany', 'N/A')} km
• Score entrevista: {candidate.get('InterviewScore', 'N/A')}/100
• Score habilidades: {candidate.get('SkillScore', 'N/A')}/100
• Score personalidad: {candidate.get('PersonalityScore', 'N/A')}/100
• Estrategia reclutamiento: {df['RecruitmentStrategy'].iloc[0] if 'RecruitmentStrategy' in df.columns else 'N/A'}

 **Información del modelo**:
• Algoritmo: Random Forest
• Accuracy: 91.67%
• F1-Score: 85.21%

 **Datos JSON procesados**:
```json
{json.dumps(candidate, indent=2, ensure_ascii=False)}
```
        """
        
        return result
        
    except json.JSONDecodeError as e:
        return f" Error al parsear JSON: {str(e)}\n\nVerifica que el archivo tenga formato JSON válido."
    except Exception as e:
        error_msg = f" Error inesperado: {str(e)}\n\nDetalles técnicos:\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

# Crear interfaz Gradio
demo = gr.Interface(
    fn=predict_hiring,
    inputs=gr.File(
        label=" Archivo JSON del candidato",
        file_types=[".json"]
    ),
    outputs=gr.Textbox(
        label=" Resultado de la predicción", 
        lines=25,
        max_lines=30
    ),
    title=" Predictor de Contratación ML",
    description="""
    **Sistema de predicción de contratación basado en Machine Learning**
    
    Sube un archivo JSON con los datos del candidato para obtener una predicción automática.
    
    **Formato esperado del JSON**:
    ```json
    [{
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
    }]
    ```
    """,
    examples=[
        ["vale_data.json"]
    ] if os.path.exists("vale_data.json") else None,
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    print("🚀 Iniciando servidor Gradio...")
    print("🌐 URL: http://localhost:7861")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False
    )