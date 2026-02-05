"""
DermaAI Vision - Flask Backend Server
=====================================

Servidor API para clasificación de enfermedades dermatológicas con
integración de modelo CNN y agente de aprendizaje por refuerzo.
"""

import os
import sys
import json
import base64
import numpy as np
from io import BytesIO
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# Configurar rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
FRONTEND_DIR = os.path.join(PROJECT_DIR, 'frontend')

# Crear la aplicación Flask
app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# Variables globales para los modelos
cnn_model = None
rl_agent = None
unet_model = None
class_info = None

# Tamaño para modelo U-Net
UNET_SIZE = (128, 128)

# ============================================
# CONFIGURACIÓN
# ============================================

# Tamaño de imagen esperado por el modelo
IMG_SIZE = (224, 224)

# Niveles de riesgo por diagnóstico
RISK_LEVELS = {
    'Melanoma, NOS': 'CRITICAL',
    'Squamous cell carcinoma, NOS': 'HIGH',
    'Basal cell carcinoma': 'HIGH',
    'Solar or actinic keratosis': 'MEDIUM',
    'Pigmented benign keratosis': 'LOW',
    'Dermatofibroma': 'LOW',
    'Nevus': 'LOW'
}

# Recomendaciones por diagnóstico
RECOMMENDATIONS = {
    'Melanoma, NOS': {
        'urgency': 'URGENTE',
        'action': 'Consulta inmediata con dermatólogo oncológico. Biopsia recomendada.',
        'color': '#e74c3c'
    },
    'Squamous cell carcinoma, NOS': {
        'urgency': 'ALTA',
        'action': 'Consulta con dermatólogo dentro de 1-2 semanas. Evaluación para posible biopsia.',
        'color': '#e67e22'
    },
    'Basal cell carcinoma': {
        'urgency': 'ALTA',
        'action': 'Consulta con dermatólogo. Usualmente tratable con cirugía menor.',
        'color': '#e67e22'
    },
    'Solar or actinic keratosis': {
        'urgency': 'MEDIA',
        'action': 'Seguimiento con dermatólogo. Lesión precancerosa que requiere monitoreo.',
        'color': '#f39c12'
    },
    'Pigmented benign keratosis': {
        'urgency': 'BAJA',
        'action': 'Lesión benigna. Control rutinario recomendado.',
        'color': '#27ae60'
    },
    'Dermatofibroma': {
        'urgency': 'BAJA',
        'action': 'Lesión benigna común. No requiere tratamiento a menos que cause molestias.',
        'color': '#27ae60'
    },
    'Nevus': {
        'urgency': 'BAJA',
        'action': 'Lunar normal. Monitorear cambios en tamaño, forma o color.',
        'color': '#27ae60'
    }
}


# ============================================
# CARGA DE MODELOS
# ============================================

def load_models():
    """Carga los modelos CNN y RL al iniciar el servidor."""
    global cnn_model, rl_agent, class_info
    
    print("🔄 Cargando modelos...")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        # Cargar modelo CNN
        cnn_path = os.path.join(MODELS_DIR, 'dermaai_model_final.keras')
        if os.path.exists(cnn_path):
            cnn_model = keras.models.load_model(cnn_path)
            print(f"✅ Modelo CNN cargado desde {cnn_path}")
        else:
            # Intentar con .h5
            cnn_path_h5 = os.path.join(MODELS_DIR, 'dermaai_model_final.h5')
            if os.path.exists(cnn_path_h5):
                cnn_model = keras.models.load_model(cnn_path_h5)
                print(f"✅ Modelo CNN cargado desde {cnn_path_h5}")
            else:
                print(f"⚠️ Modelo CNN no encontrado. Usando modo demo.")
        
        # Cargar información de clases
        class_info_path = os.path.join(MODELS_DIR, 'class_info.json')
        if os.path.exists(class_info_path):
            with open(class_info_path, 'r') as f:
                class_info = json.load(f)
            print(f"✅ Información de clases cargada")
        else:
            # Usar clases por defecto
            class_info = {
                'class_names': [
                    'Basal cell carcinoma',
                    'Dermatofibroma',
                    'Melanoma, NOS',
                    'Nevus',
                    'Pigmented benign keratosis',
                    'Solar or actinic keratosis',
                    'Squamous cell carcinoma, NOS'
                ],
                'num_classes': 7
            }
            print(f"⚠️ Usando clases por defecto")
        
        # Cargar agente RL
        rl_model_path = os.path.join(MODELS_DIR, 'escalation_agent_model.keras')
        if os.path.exists(rl_model_path):
            rl_agent = keras.models.load_model(rl_model_path)
            print(f"✅ Agente RL cargado")
        else:
            print(f"⚠️ Agente RL no encontrado. Usando reglas heurísticas.")
        
        # Cargar modelo U-Net para segmentación
        global unet_model
        unet_path = os.path.join(MODELS_DIR, 'unet_segmentation_model.keras')
        if os.path.exists(unet_path):
            unet_model = keras.models.load_model(unet_path, compile=False)
            print(f"✅ Modelo U-Net cargado")
        else:
            print(f"⚠️ Modelo U-Net no encontrado. Mejora de imágenes no disponible.")
        
        print("✅ Modelos listos!")
        return True
        
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        return False


def preprocess_image(image):
    """Preprocesa una imagen para el modelo CNN."""
    # Redimensionar
    image = image.resize(IMG_SIZE)
    
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convertir a array y normalizar
    img_array = np.array(image) / 255.0
    
    # Agregar dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def get_rl_state(prediction_probs):
    """Construye el vector de estado para el agente RL."""
    max_conf = np.max(prediction_probs)
    entropy = -np.sum(prediction_probs * np.log(prediction_probs + 1e-10))
    normalized_entropy = entropy / np.log(len(prediction_probs))
    
    state = np.concatenate([
        prediction_probs,
        [max_conf],
        [normalized_entropy]
    ])
    return state.reshape(1, -1)


def decide_escalation_heuristic(prediction_probs, class_names):
    """Decisión de escalamiento usando reglas heurísticas (fallback)."""
    max_idx = np.argmax(prediction_probs)
    confidence = prediction_probs[max_idx]
    class_name = class_names[max_idx]
    risk = RISK_LEVELS.get(class_name, 'MEDIUM')
    
    # Reglas de escalamiento
    should_escalate = (
        risk == 'CRITICAL' or
        (risk == 'HIGH' and confidence < 0.85) or
        (risk == 'MEDIUM' and confidence < 0.60) or
        confidence < 0.50
    )
    
    return should_escalate


# ============================================
# RUTAS API
# ============================================

@app.route('/')
def serve_frontend():
    """Sirve la página principal."""
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Sirve archivos estáticos del frontend."""
    return send_from_directory(FRONTEND_DIR, path)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Verifica el estado del servidor."""
    return jsonify({
        'status': 'ok',
        'cnn_model_loaded': cnn_model is not None,
        'rl_agent_loaded': rl_agent is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Retorna la lista de clases disponibles."""
    if class_info:
        return jsonify({
            'classes': class_info['class_names'],
            'num_classes': class_info['num_classes'],
            'risk_levels': RISK_LEVELS
        })
    return jsonify({'error': 'Información de clases no disponible'}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Realiza predicción sobre una imagen.
    
    Acepta:
        - image_base64: Imagen en formato base64
        - image_file: Archivo de imagen
    
    Retorna:
        - predictions: Probabilidades por clase
        - top_prediction: Clase con mayor probabilidad
        - confidence: Nivel de confianza
        - should_escalate: Decisión del agente RL
        - recommendation: Recomendación médica
    """
    try:
        # Obtener imagen
        if 'image_base64' in request.json:
            # Decodificar base64
            image_data = request.json['image_base64']
            # Remover prefijo data:image si existe
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
        
        elif 'image' in request.files:
            # Archivo subido
            image_file = request.files['image']
            image = Image.open(image_file.stream)
        
        else:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        # Preprocesar imagen
        processed_image = preprocess_image(image)
        
        # Realizar predicción
        if cnn_model is not None:
            predictions = cnn_model.predict(processed_image, verbose=0)[0]
        else:
            # Modo demo: predicción aleatoria
            predictions = np.random.dirichlet(np.ones(7))
        
        # Obtener clase con mayor probabilidad
        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx])
        class_names = class_info['class_names'] if class_info else list(RISK_LEVELS.keys())
        top_class = class_names[top_idx]
        
        # Decisión de escalamiento
        if rl_agent is not None:
            state = get_rl_state(predictions)
            q_values = rl_agent.predict(state, verbose=0)[0]
            should_escalate = int(np.argmax(q_values)) == 1
        else:
            should_escalate = decide_escalation_heuristic(predictions, class_names)
        
        # Obtener recomendación
        recommendation = RECOMMENDATIONS.get(top_class, {
            'urgency': 'MEDIA',
            'action': 'Consulte con un dermatólogo para evaluación.',
            'color': '#f39c12'
        })
        
        # Preparar respuesta
        response = {
            'success': True,
            'predictions': {
                class_names[i]: float(predictions[i]) 
                for i in range(len(predictions))
            },
            'top_prediction': {
                'class': top_class,
                'confidence': confidence,
                'risk_level': RISK_LEVELS.get(top_class, 'UNKNOWN')
            },
            'escalation': {
                'should_escalate': should_escalate,
                'reason': 'Alto riesgo o baja confianza' if should_escalate else 'Caso manejable',
                'agent_type': 'DQN' if rl_agent else 'Heurístico'
            },
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================
# FUNCIONES DE MEJORA DE IMAGEN
# ============================================

def remove_shadows(img_array):
    """Elimina sombras usando filtrado homomórfico."""
    import cv2
    img = (img_array * 255).astype(np.uint8)
    
    # Convertir a float y aplicar log
    img_float = img.astype(np.float64) + 1
    img_log = np.log(img_float)
    
    # Aplicar filtro Gaussiano en cada canal
    result = np.zeros_like(img_float)
    for i in range(3):
        low_freq = cv2.GaussianBlur(img_log[:,:,i], (21, 21), 0)
        high_freq = img_log[:,:,i] - low_freq
        result[:,:,i] = np.exp(high_freq + np.mean(low_freq))
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result / 255.0


def enhance_contrast(img_array):
    """Mejora el contraste usando CLAHE."""
    import cv2
    img = (img_array * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return enhanced / 255.0


@app.route('/api/enhance', methods=['POST'])
def enhance_image():
    """
    Mejora una imagen usando segmentación U-Net y técnicas de procesamiento.
    
    Acepta:
        - image_base64: Imagen en formato base64
    
    Retorna:
        - original: Imagen original en base64
        - enhanced: Imagen mejorada en base64
        - mask: Máscara de segmentación en base64
        - segmented: Imagen segmentada en base64
    """
    try:
        import cv2
        
        # Obtener imagen
        if 'image_base64' not in request.json:
            return jsonify({'error': 'No se proporcionó imagen'}), 400
        
        image_data = request.json['image_base64']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convertir a RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Redimensionar para U-Net
        original_size = image.size
        img_unet = image.resize(UNET_SIZE)
        img_array = np.array(img_unet) / 255.0
        
        # Inicializar variables de salida
        mask_binary = None
        segmented_array = None
        
        # Generar máscara con U-Net si está disponible
        if unet_model is not None:
            img_batch = np.expand_dims(img_array, axis=0)
            mask = unet_model.predict(img_batch, verbose=0)[0].squeeze()
            mask_binary = (mask > 0.5).astype(np.float32)
        else:
            # Fallback: usar Otsu thresholding
            gray = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mask_binary = thresh / 255.0
        
        # Aplicar mejoras
        img_no_shadows = remove_shadows(img_array)
        img_enhanced = enhance_contrast(img_no_shadows)
        
        # Aplicar máscara
        mask_3d = np.expand_dims(mask_binary, axis=-1)
        segmented_array = img_enhanced * mask_3d
        
        # Convertir a base64
        def array_to_base64(arr):
            arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            img_pil = Image.fromarray(arr_uint8)
            img_pil = img_pil.resize(original_size)
            buffer = BytesIO()
            img_pil.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        def mask_to_base64(m):
            m_uint8 = (m * 255).astype(np.uint8)
            img_pil = Image.fromarray(m_uint8)
            img_pil = img_pil.resize(original_size)
            buffer = BytesIO()
            img_pil.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        response = {
            'success': True,
            'original': array_to_base64(img_array),
            'enhanced': array_to_base64(img_enhanced),
            'mask': mask_to_base64(mask_binary),
            'segmented': array_to_base64(segmented_array),
            'unet_available': unet_model is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error en mejora de imagen: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Retorna información sobre los modelos cargados."""
    info = {
        'cnn_model': {
            'loaded': cnn_model is not None,
            'type': 'MobileNetV2 Transfer Learning',
            'input_size': list(IMG_SIZE) + [3],
            'num_classes': class_info['num_classes'] if class_info else 7
        },
        'rl_agent': {
            'loaded': rl_agent is not None,
            'type': 'DQN (Deep Q-Network)',
            'actions': ['No escalar', 'Escalar a especialista']
        },
        'risk_levels': RISK_LEVELS
    }
    return jsonify(info)


# ============================================
# INICIO DEL SERVIDOR
# ============================================

if __name__ == '__main__':
    # Crear carpeta de modelos si no existe
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Cargar modelos
    load_models()
    
    # Iniciar servidor
    print("\n" + "="*50)
    print("🚀 DermaAI Vision Server")
    print("="*50)
    print(f"📁 Frontend: {FRONTEND_DIR}")
    print(f"📁 Modelos: {MODELS_DIR}")
    print(f"🌐 URL: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
