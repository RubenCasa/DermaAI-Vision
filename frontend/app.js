/**
 * DermaAI Vision - Frontend JavaScript
 * =====================================
 * 
 * Handles camera capture, file upload, API communication,
 * and dynamic UI updates for diagnosis results.
 */

// ============================================
// CONFIGURATION
// ============================================

const API_BASE_URL = 'http://localhost:5000';

// State
let cameraStream = null;
let currentImageData = null;
let isAnalyzing = false;

// DOM Elements
const elements = {
    // Server status
    serverStatus: document.getElementById('serverStatus'),

    // Tabs
    tabButtons: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),

    // Camera
    cameraPreview: document.getElementById('cameraPreview'),
    cameraCanvas: document.getElementById('cameraCanvas'),
    cameraOverlay: document.getElementById('cameraOverlay'),
    btnStartCamera: document.getElementById('btnStartCamera'),
    btnCapture: document.getElementById('btnCapture'),

    // Upload
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    previewContainer: document.getElementById('previewContainer'),
    previewImage: document.getElementById('previewImage'),
    btnClearPreview: document.getElementById('btnClearPreview'),

    // Analysis
    btnAnalyze: document.getElementById('btnAnalyze'),

    // Results
    resultsPlaceholder: document.getElementById('resultsPlaceholder'),
    resultsContent: document.getElementById('resultsContent'),
    analysisTimestamp: document.getElementById('analysisTimestamp'),

    // Diagnosis
    diagnosisCard: document.getElementById('diagnosisCard'),
    diagnosisName: document.getElementById('diagnosisName'),
    diagnosisConfidence: document.getElementById('diagnosisConfidence'),
    diagnosisRisk: document.getElementById('diagnosisRisk'),

    // Escalation
    escalationCard: document.getElementById('escalationCard'),
    escalationIcon: document.getElementById('escalationIcon'),
    escalationDecision: document.getElementById('escalationDecision'),
    escalationReason: document.getElementById('escalationReason'),

    // Recommendation
    recommendationCard: document.getElementById('recommendationCard'),
    recommendationUrgency: document.getElementById('recommendationUrgency'),
    recommendationText: document.getElementById('recommendationText'),

    // Probabilities
    probabilitiesList: document.getElementById('probabilitiesList'),

    // Toast
    toastContainer: document.getElementById('toastContainer')
};


// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initCamera();
    initUpload();
    initAnalyze();
    checkServerHealth();

    // Check server health periodically
    setInterval(checkServerHealth, 30000);
});


// ============================================
// SERVER HEALTH CHECK
// ============================================

async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();

        updateServerStatus(data.status === 'ok', data);
    } catch (error) {
        updateServerStatus(false);
    }
}

function updateServerStatus(isConnected, data = null) {
    const statusDot = elements.serverStatus.querySelector('.status-dot');
    const statusText = elements.serverStatus.querySelector('.status-text');

    if (isConnected) {
        statusDot.className = 'status-dot connected';
        const modelStatus = data?.cnn_model_loaded ? '✓ Modelo' : '⚠ Demo';
        statusText.textContent = `Conectado ${modelStatus}`;
    } else {
        statusDot.className = 'status-dot error';
        statusText.textContent = 'Sin conexión';
    }
}


// ============================================
// TABS FUNCTIONALITY
// ============================================

function initTabs() {
    elements.tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update buttons
    elements.tabButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    // Update content
    elements.tabContents.forEach(content => {
        content.classList.toggle('active', content.id === `tab-${tabName}`);
    });

    // Reset current image when switching tabs
    if (tabName === 'camera') {
        clearUploadPreview();
    } else {
        stopCamera();
    }
}


// ============================================
// CAMERA FUNCTIONALITY
// ============================================

function initCamera() {
    elements.btnStartCamera.addEventListener('click', toggleCamera);
    elements.btnCapture.addEventListener('click', captureImage);
}

async function toggleCamera() {
    if (cameraStream) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    try {
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment' // Prefer back camera on mobile
            }
        };

        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.cameraPreview.srcObject = cameraStream;

        elements.cameraOverlay.classList.add('hidden');
        elements.btnStartCamera.innerHTML = '<span>⏹️</span> Detener Cámara';
        elements.btnCapture.disabled = false;

        showToast('Cámara iniciada', 'success');
    } catch (error) {
        console.error('Error accessing camera:', error);
        showToast('No se pudo acceder a la cámara', 'error');
        elements.cameraOverlay.querySelector('span').textContent = 'Error al acceder a la cámara';
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    elements.cameraPreview.srcObject = null;
    elements.cameraOverlay.classList.remove('hidden');
    elements.cameraOverlay.querySelector('span').textContent = 'Cámara no iniciada';
    elements.btnStartCamera.innerHTML = '<span>🎥</span> Iniciar Cámara';
    elements.btnCapture.disabled = true;
}

function captureImage() {
    if (!cameraStream) return;

    const video = elements.cameraPreview;
    const canvas = elements.cameraCanvas;
    const ctx = canvas.getContext('2d');

    // Set canvas size to video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0);

    // Get image data as base64
    currentImageData = canvas.toDataURL('image/jpeg', 0.9);

    // Enable analyze button
    elements.btnAnalyze.disabled = false;

    // Visual feedback
    elements.cameraPreview.style.animation = 'flash 0.3s ease';
    setTimeout(() => {
        elements.cameraPreview.style.animation = '';
    }, 300);

    showToast('Imagen capturada', 'success');
}


// ============================================
// FILE UPLOAD FUNCTIONALITY
// ============================================

function initUpload() {
    // Click to upload
    elements.uploadZone.addEventListener('click', () => {
        elements.fileInput.click();
    });

    // File selected
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    elements.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadZone.classList.add('dragover');
    });

    elements.uploadZone.addEventListener('dragleave', () => {
        elements.uploadZone.classList.remove('dragover');
    });

    elements.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadZone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Clear preview
    elements.btnClearPreview.addEventListener('click', clearUploadPreview);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showToast('Por favor selecciona una imagen', 'error');
        return;
    }

    // Read file
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImageData = e.target.result;

        // Show preview
        elements.previewImage.src = currentImageData;
        elements.uploadZone.style.display = 'none';
        elements.previewContainer.style.display = 'block';

        // Enable analyze button
        elements.btnAnalyze.disabled = false;

        showToast('Imagen cargada', 'success');
    };

    reader.readAsDataURL(file);
}

function clearUploadPreview() {
    currentImageData = null;
    elements.fileInput.value = '';
    elements.previewImage.src = '';
    elements.uploadZone.style.display = 'block';
    elements.previewContainer.style.display = 'none';
    elements.btnAnalyze.disabled = true;
}


// ============================================
// ANALYSIS FUNCTIONALITY
// ============================================

function initAnalyze() {
    elements.btnAnalyze.addEventListener('click', analyzeImage);
}

async function analyzeImage() {
    if (!currentImageData || isAnalyzing) return;

    isAnalyzing = true;
    setAnalyzeButtonLoading(true);

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_base64: currentImageData
            })
        });

        if (!response.ok) {
            throw new Error('Error en la predicción');
        }

        const result = await response.json();

        if (result.success) {
            displayResults(result);
            showToast('Análisis completado', 'success');
        } else {
            throw new Error(result.error || 'Error desconocido');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showToast(`Error: ${error.message}`, 'error');
    } finally {
        isAnalyzing = false;
        setAnalyzeButtonLoading(false);
    }
}

function setAnalyzeButtonLoading(loading) {
    const btnText = elements.btnAnalyze.querySelector('.btn-text');
    const btnLoader = elements.btnAnalyze.querySelector('.btn-loader');

    if (loading) {
        btnText.style.visibility = 'hidden';
        btnLoader.style.display = 'flex';
        elements.btnAnalyze.disabled = true;
    } else {
        btnText.style.visibility = 'visible';
        btnLoader.style.display = 'none';
        elements.btnAnalyze.disabled = !currentImageData;
    }
}


// ============================================
// RESULTS DISPLAY
// ============================================

function displayResults(result) {
    // Hide placeholder, show results
    elements.resultsPlaceholder.style.display = 'none';
    elements.resultsContent.style.display = 'flex';

    // Update timestamp
    const timestamp = new Date(result.timestamp);
    elements.analysisTimestamp.textContent = `Análisis realizado: ${timestamp.toLocaleString()}`;

    // Display diagnosis
    displayDiagnosis(result.top_prediction);

    // Display escalation decision
    displayEscalation(result.escalation);

    // Display recommendation
    displayRecommendation(result.recommendation);

    // Display probabilities
    displayProbabilities(result.predictions);

    // Animate cards
    animateResults();
}

function displayDiagnosis(prediction) {
    elements.diagnosisName.textContent = prediction.class;
    elements.diagnosisConfidence.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;

    // Risk badge
    const riskLevel = prediction.risk_level.toLowerCase();
    const riskBadge = elements.diagnosisRisk.querySelector('.risk-badge');
    riskBadge.textContent = prediction.risk_level;
    riskBadge.className = `risk-badge ${riskLevel}`;

    // Update card border color based on risk
    const colors = {
        critical: '#ef4444',
        high: '#f97316',
        medium: '#eab308',
        low: '#22c55e'
    };
    elements.diagnosisCard.style.borderColor = colors[riskLevel] || colors.medium;
}

function displayEscalation(escalation) {
    const shouldEscalate = escalation.should_escalate;

    elements.escalationCard.className = `escalation-card ${shouldEscalate ? 'escalate' : 'no-escalate'}`;
    elements.escalationIcon.textContent = shouldEscalate ? '🚨' : '✅';

    const decisionText = elements.escalationDecision.querySelector('.decision-text');
    decisionText.textContent = shouldEscalate
        ? 'ESCALAR A ESPECIALISTA'
        : 'CASO MANEJABLE';

    elements.escalationReason.textContent = `${escalation.reason} (Agente: ${escalation.agent_type})`;
}

function displayRecommendation(recommendation) {
    const urgencyColors = {
        'URGENTE': '#ef4444',
        'ALTA': '#f97316',
        'MEDIA': '#eab308',
        'BAJA': '#22c55e'
    };

    elements.recommendationUrgency.textContent = recommendation.urgency;
    elements.recommendationUrgency.style.background = recommendation.color || urgencyColors[recommendation.urgency];
    elements.recommendationText.textContent = recommendation.action;
    elements.recommendationCard.style.borderLeftColor = recommendation.color || urgencyColors[recommendation.urgency];
}

function displayProbabilities(predictions) {
    // Sort by probability
    const sorted = Object.entries(predictions)
        .sort((a, b) => b[1] - a[1]);

    elements.probabilitiesList.innerHTML = sorted.map(([className, prob]) => `
        <div class="probability-item">
            <div class="probability-bar-container">
                <span class="probability-label" title="${className}">${className}</span>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: 0%;" data-width="${prob * 100}%"></div>
                </div>
            </div>
            <span class="probability-value">${(prob * 100).toFixed(1)}%</span>
        </div>
    `).join('');

    // Animate bars
    requestAnimationFrame(() => {
        setTimeout(() => {
            document.querySelectorAll('.probability-fill').forEach(bar => {
                bar.style.width = bar.dataset.width;
            });
        }, 100);
    });
}

function animateResults() {
    const cards = elements.resultsContent.querySelectorAll('.diagnosis-card, .escalation-card, .recommendation-card, .probabilities-card');

    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';

        setTimeout(() => {
            card.style.transition = 'all 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
}


// ============================================
// TOAST NOTIFICATIONS
// ============================================

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };

    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${message}</span>
    `;

    elements.toastContainer.appendChild(toast);

    // Remove after animation
    setTimeout(() => {
        toast.remove();
    }, 5000);
}


// ============================================
// UTILITY STYLES (added via JS)
// ============================================

// Add flash animation for capture
const style = document.createElement('style');
style.textContent = `
    @keyframes flash {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.5); }
    }
`;
document.head.appendChild(style);
