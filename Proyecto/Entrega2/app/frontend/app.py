"""
Frontend Gradio - SodAI Drinks Recommendation System
===================================================

Interfaz de usuario interactiva para el sistema de recomendación
de productos usando Gradio.

Autor: SodAI Drinks MLOps Team
"""

import gradio as gr
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la API
API_BASE_URL = os.getenv('API_BASE_URL', 'http://backend:8000')

# Estado global para historial
prediction_history = []

def check_api_health() -> Dict[str, Any]:
    """Verifica el estado de la API backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        logger.error(f"Error conectando con API: {e}")
        return {"status": "unhealthy", "error": str(e)}

def predict_single_product(customer_id: int, product_id: int) -> Tuple[str, str, str]:
    """
    Realiza una predicción individual y retorna información detallada.
    
    Returns:
        Tuple de (resultado_predicción, info_cliente, info_producto)
    """
    try:
        # Predicción
        prediction_data = {
            "customer_id": customer_id,
            "product_id": product_id
        }
        
        pred_response = requests.post(
            f"{API_BASE_URL}/predict",
            json=prediction_data,
            timeout=10
        )
        
        if pred_response.status_code == 200:
            prediction = pred_response.json()
            
            # Información del cliente
            try:
                customer_response = requests.get(f"{API_BASE_URL}/customers/{customer_id}/info")
                customer_info = customer_response.json() if customer_response.status_code == 200 else {}
            except:
                customer_info = {}
            
            # Información del producto
            try:
                product_response = requests.get(f"{API_BASE_URL}/products/{product_id}/info")
                product_info = product_response.json() if product_response.status_code == 200 else {}
            except:
                product_info = {}
            
            # Guardar en historial
            prediction_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'customer_id': customer_id,
                'product_id': product_id,
                'prediction_score': prediction['prediction_score'],
                'confidence': prediction['recommendation_confidence']
            })
            
            # Formatear resultados
            score = prediction['prediction_score']
            confidence = prediction['recommendation_confidence']
            
            # Resultado de predicción
            prediction_result = f"""
            ## 🎯 Resultado de Predicción
            
            **Probabilidad de Compra:** {score:.1%}
            
            **Nivel de Confianza:** {confidence.upper()}
            
            **Interpretación:**
            {'🟢 Alta probabilidad - Recomendación fuerte' if score >= 0.7 
             else '🟡 Probabilidad media - Recomendación moderada' if score >= 0.4 
             else '🔴 Baja probabilidad - No recomendado'}
            
            **Timestamp:** {prediction['timestamp']}
            """
            
            # Información del cliente
            customer_result = f"""
            ## 👤 Información del Cliente
            
            **ID Cliente:** {customer_id}
            **Región:** {customer_info.get('region_id', 'N/A')}
            **Zona:** {customer_info.get('zone_id', 'N/A')}
            **Tipo:** {customer_info.get('customer_type', 'N/A')}
            **Ubicación:** ({customer_info.get('location_x', 'N/A')}, {customer_info.get('location_y', 'N/A')})
            """
            
            # Información del producto
            product_result = f"""
            ## 🥤 Información del Producto
            
            **ID Producto:** {product_id}
            **Marca:** {product_info.get('brand', 'N/A')}
            **Categoría:** {product_info.get('category', 'N/A')}
            **Subcategoría:** {product_info.get('sub_category', 'N/A')}
            **Segmento:** {product_info.get('segment', 'N/A')}
            **Tamaño:** {product_info.get('size', 'N/A')}
            **Empaque:** {product_info.get('package', 'N/A')}
            """
            
            return prediction_result, customer_result, product_result
            
        else:
            error_msg = f"Error en predicción: {pred_response.status_code}"
            return error_msg, "Error cargando información", "Error cargando información"
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, "Error cargando información", "Error cargando información"

def get_customer_recommendations(customer_id: int, top_k: int = 10) -> Tuple[str, str]:
    """
    Obtiene recomendaciones para un cliente específico.
    
    Returns:
        Tuple de (tabla_recomendaciones, gráfico_scores)
    """
    try:
        recommendation_data = {
            "customer_id": customer_id,
            "top_k": top_k
        }
        
        response = requests.post(
            f"{API_BASE_URL}/recommendations",
            json=recommendation_data,
            timeout=15
        )
        
        if response.status_code == 200:
            recommendations = response.json()
            
            if recommendations['recommendations']:
                # Crear DataFrame para mostrar
                df = pd.DataFrame(recommendations['recommendations'])
                
                # Formatear tabla
                df['prediction_score'] = df['prediction_score'].apply(lambda x: f"{x:.1%}")
                df_display = df[['product_id', 'product_brand', 'product_category', 
                               'product_segment', 'prediction_score']].copy()
                df_display.columns = ['ID Producto', 'Marca', 'Categoría', 'Segmento', 'Probabilidad']
                
                # Crear gráfico
                fig = px.bar(
                    df.head(10), 
                    x='product_id', 
                    y='prediction_score',
                    color='product_category',
                    title=f'Top {min(top_k, 10)} Recomendaciones para Cliente {customer_id}',
                    labels={'prediction_score': 'Probabilidad de Compra', 'product_id': 'ID Producto'}
                )
                fig.update_layout(showlegend=True, height=400)
                
                return df_display.to_html(index=False, classes='table table-striped'), fig
            else:
                return "No se encontraron recomendaciones", None
                
        else:
            return f"Error obteniendo recomendaciones: {response.status_code}", None
            
    except Exception as e:
        return f"Error: {str(e)}", None

def show_prediction_history() -> Tuple[str, Any]:
    """
    Muestra el historial de predicciones.
    
    Returns:
        Tuple de (tabla_historial, gráfico_tendencias)
    """
    if not prediction_history:
        return "No hay predicciones en el historial", None
    
    # Crear DataFrame del historial
    df_history = pd.DataFrame(prediction_history)
    df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
    
    # Formatear para mostrar
    df_display = df_history.copy()
    df_display['prediction_score'] = df_display['prediction_score'].apply(lambda x: f"{x:.1%}")
    df_display.columns = ['Tiempo', 'Cliente', 'Producto', 'Probabilidad', 'Confianza']
    
    # Gráfico de tendencias
    fig = px.line(
        df_history, 
        x='timestamp', 
        y='prediction_score',
        color='confidence',
        title='Tendencia de Scores de Predicción',
        labels={'prediction_score': 'Score de Predicción', 'timestamp': 'Tiempo'}
    )
    fig.update_layout(height=300)
    
    return df_display.to_html(index=False, classes='table table-striped'), fig

def clear_history() -> str:
    """Limpia el historial de predicciones."""
    global prediction_history
    prediction_history = []
    return "Historial limpiado exitosamente"

# Crear interfaz Gradio
def create_gradio_app():
    """Crea la aplicación Gradio."""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .tab-nav button {
        font-weight: bold;
    }
    .table {
        font-size: 12px;
    }
    """
    
    with gr.Blocks(css=custom_css, title="SodAI Drinks - Sistema de Recomendación") as app:
        
        # Título principal
        gr.Markdown("""
        # 🥤 SodAI Drinks - Sistema de Recomendación de Productos
        
        **Sistema MLOps para recomendación inteligente de bebidas**
        
        Utiliza machine learning para predecir qué productos es más probable que compre cada cliente.
        """)
        
        # Check de estado de la API
        with gr.Row():
            api_status = gr.Textbox(
                label="Estado de la API",
                value="Verificando...",
                interactive=False
            )
            refresh_btn = gr.Button("🔄 Actualizar Estado", size="sm")
        
        # Tabs principales
        with gr.Tabs():
            
            # Tab 1: Predicción Individual
            with gr.TabItem("🎯 Predicción Individual"):
                gr.Markdown("### Predice la probabilidad de que un cliente específico compre un producto")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        customer_id_input = gr.Number(
                            label="ID del Cliente",
                            value=256017,
                            precision=0,
                            info="Ingresa el ID del cliente"
                        )
                        product_id_input = gr.Number(
                            label="ID del Producto",
                            value=34092,
                            precision=0,
                            info="Ingresa el ID del producto"
                        )
                        predict_btn = gr.Button("🔮 Predecir", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        prediction_output = gr.Markdown(label="Resultado de Predicción")
                
                with gr.Row():
                    with gr.Column():
                        customer_info_output = gr.Markdown(label="Información del Cliente")
                    with gr.Column():
                        product_info_output = gr.Markdown(label="Información del Producto")
            
            # Tab 2: Recomendaciones por Cliente
            with gr.TabItem("📋 Recomendaciones por Cliente"):
                gr.Markdown("### Obtén las mejores recomendaciones para un cliente específico")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        rec_customer_id = gr.Number(
                            label="ID del Cliente",
                            value=256017,
                            precision=0
                        )
                        rec_top_k = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=10,
                            step=5,
                            label="Número de Recomendaciones"
                        )
                        get_rec_btn = gr.Button("📊 Obtener Recomendaciones", variant="primary")
                    
                    with gr.Column(scale=2):
                        recommendations_output = gr.HTML(label="Top Recomendaciones")
                
                recommendations_plot = gr.Plot(label="Gráfico de Recomendaciones")
            
            # Tab 3: Historial y Analytics
            with gr.TabItem("📈 Historial & Analytics"):
                gr.Markdown("### Historial de predicciones y análisis de tendencias")
                
                with gr.Row():
                    show_history_btn = gr.Button("📋 Mostrar Historial", variant="secondary")
                    clear_history_btn = gr.Button("🗑️ Limpiar Historial", variant="stop")
                
                history_output = gr.HTML(label="Historial de Predicciones")
                history_plot = gr.Plot(label="Tendencias de Predicciones")
                clear_status = gr.Textbox(label="Estado", visible=False)
        
        # Información adicional
        with gr.Accordion("ℹ️ Información del Sistema", open=False):
            gr.Markdown("""
            ### Cómo usar el sistema:
            
            1. **Predicción Individual**: Ingresa un ID de cliente y producto para obtener la probabilidad de compra
            2. **Recomendaciones**: Ingresa un ID de cliente para ver los productos más recomendados
            3. **Historial**: Revisa las predicciones anteriores y sus tendencias
            
            ### Interpretación de resultados:
            
            - **🟢 Alta (>70%)**: Recomendación fuerte
            - **🟡 Media (40-70%)**: Recomendación moderada  
            - **🔴 Baja (<40%)**: No recomendado
            
            ### Datos de ejemplo:
            - **Clientes**: 256017, 255780, 254655, 254445, 254403
            - **Productos**: 34092, 57290, 56714, 296616, 60854
            """)
        
        # Event handlers
        def update_api_status():
            health = check_api_health()
            if health.get("status") == "healthy":
                return "🟢 API Conectada - Modelo Cargado ✅"
            else:
                return f"🔴 API Error: {health.get('error', 'Unknown error')}"
        
        # Conectar eventos
        refresh_btn.click(fn=update_api_status, outputs=api_status)
        
        predict_btn.click(
            fn=predict_single_product,
            inputs=[customer_id_input, product_id_input],
            outputs=[prediction_output, customer_info_output, product_info_output]
        )
        
        get_rec_btn.click(
            fn=get_customer_recommendations,
            inputs=[rec_customer_id, rec_top_k],
            outputs=[recommendations_output, recommendations_plot]
        )
        
        show_history_btn.click(
            fn=show_prediction_history,
            outputs=[history_output, history_plot]
        )
        
        clear_history_btn.click(
            fn=clear_history,
            outputs=clear_status
        )
        
        # Actualizar estado inicial de la API
        app.load(fn=update_api_status, outputs=api_status)
    
    return app

if __name__ == "__main__":
    # Crear y lanzar aplicación
    app = create_gradio_app()
    
    # Configurar puerto y host
    port = int(os.getenv('PORT', 7860))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"Iniciando SodAI Drinks Frontend en {host}:{port}")
    
    app.launch(
        server_name=host,
        server_port=port,
        share=False,
        show_error=True
    )