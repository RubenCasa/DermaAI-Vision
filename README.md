# 🧠 DermaAI Vision

Sistema de clasificación de enfermedades dermatológicas con **Inteligencia Artificial** y **Aprendizaje por Refuerzo** para decisiones de escalamiento automático.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0+-green?logo=flask)

---

## 🎯 Características

| Componente | Descripción |
|------------|-------------|
| 🔬 **CNN MobileNetV2** | Clasificación de 7 condiciones cutáneas con transfer learning |
| 🤖 **Agente DQN** | Decide automáticamente cuándo escalar un caso a especialista |
| 📷 **Cámara en vivo** | Captura imágenes directamente desde el navegador |
| 📁 **Drag & Drop** | Sube imágenes arrastrándolas a la interfaz |
| ✨ **UI Moderna** | Glassmorphism, animaciones y tema oscuro premium |

---

## 📁 Estructura del Proyecto

```
PRO_ML/
├── notebooks/
│   ├── 01_Train_CNN_Model.ipynb    # Entrenar modelo CNN (Colab)
│   └── 02_Train_RL_Agent.ipynb     # Entrenar agente RL (Colab)
├── backend/
│   └── app.py                       # Servidor Flask API
├── frontend/
│   ├── index.html                   # Interfaz web
│   ├── styles.css                   # Estilos CSS
│   └── app.js                       # JavaScript
├── models/                          # Modelos entrenados
├── requirements.txt                 # Dependencias Python
└── README.md
```

---

## 🚀 Inicio Rápido

### 1️⃣ Entrenar Modelos (Google Colab)

1. Abre `notebooks/01_Train_CNN_Model.ipynb` en [Google Colab](https://colab.research.google.com)
2. Ejecuta todas las celdas (el dataset se descarga automáticamente de Kaggle)
3. Descarga los archivos generados:
   - `dermaai_model_final.keras`
   - `class_info.json`
4. Abre `notebooks/02_Train_RL_Agent.ipynb` y repite el proceso
5. Descarga:
   - `escalation_agent_model.keras`
   - `escalation_agent_config.json`

### 2️⃣ Configurar Localmente

```powershell
# Instalar dependencias
cd c:\Users\Desktop
pip install -r requirements.txt

# Copiar modelos entrenados a la carpeta models/
# (los archivos descargados de Colab)
```

### 3️⃣ Ejecutar la Aplicación

```powershell
python backend/app.py
```

Abre en tu navegador: **http://localhost:5000**

---

## 🖼️ Uso

1. **Cámara**: Inicia la cámara y captura una imagen de la lesión cutánea
2. **Upload**: Arrastra una imagen o haz clic para seleccionar
3. **Analizar**: Haz clic en "Analizar Imagen"
4. **Resultados**: Verás el diagnóstico, nivel de riesgo, y decisión del agente RL

---

## 📊 Clases de Diagnóstico

| Condición | Nivel de Riesgo |
|-----------|----------------|
| 🔴 Melanoma | CRÍTICO |
| 🟠 Carcinoma espinocelular | ALTO |
| 🟠 Carcinoma basocelular | ALTO |
| 🟡 Queratosis actínica | MEDIO |
| 🟢 Queratosis benigna pigmentada | BAJO |
| 🟢 Dermatofibroma | BAJO |
| 🟢 Nevus (lunar) | BAJO |

---

## ⚠️ Aviso Importante

> Este sistema es **experimental** y no reemplaza la consulta con un dermatólogo profesional.
> Siempre busque atención médica especializada para el diagnóstico y tratamiento de condiciones cutáneas.

---

## 🛠️ Tecnologías

- **Backend**: Flask, TensorFlow, NumPy
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript (WebRTC)
- **ML**: MobileNetV2 (Transfer Learning), DQN (Reinforcement Learning)
- **Dataset**: ISIC Skin Cancer Dataset

---

## 📄 Licencia

Proyecto educativo desarrollado para PRO_ML.
