// Variáveis globais
let canvas, ctx, previewCanvas, previewCtx;
let isDrawing = false;
let currentTool = 'pen'; // pen or eraser
let neuralNetwork = null;
let probabilitiesList;

const GRID_SIZE = 28;
const OUTPUT_CLASSES = 10;
const HIDDEN_LAYER_SIZES = [256, 128];
const DATA_MEAN = 0.1307;
const DATA_STD = 0.3081;

// Inicialização
document.addEventListener('DOMContentLoaded', () => {
    initializeCanvas();
    setupEventListeners();
    probabilitiesList = document.getElementById('probabilitiesList');
    updateProbabilities([]);
    loadNeuralNetwork();
});

// Inicializar canvas
function initializeCanvas() {
    canvas = document.getElementById('drawingCanvas');
    ctx = canvas.getContext('2d');
    
    previewCanvas = document.getElementById('previewCanvas');
    previewCtx = previewCanvas.getContext('2d');
    
    // Configurar contexto do canvas principal
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

// Configurar event listeners
function setupEventListeners() {
    // Botões de ferramentas
    document.getElementById('penBtn').addEventListener('click', () => {
        currentTool = 'pen';
        document.getElementById('penBtn').classList.add('active');
        document.getElementById('eraserBtn').classList.remove('active');
        canvas.style.cursor = 'crosshair';
    });
    
    document.getElementById('eraserBtn').addEventListener('click', () => {
        currentTool = 'eraser';
        document.getElementById('eraserBtn').classList.add('active');
        document.getElementById('penBtn').classList.remove('active');
        canvas.style.cursor = 'grab';
    });
    
    // Botão limpar
    document.getElementById('clearBtn').addEventListener('click', clearCanvas);
    
    // Botão identificar
    document.getElementById('predictBtn').addEventListener('click', predict);
    
    // Eventos do canvas
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);
    
    // Suporte para touch (mobile)
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
}

// Funções de desenho
function getCanvasCoordinates(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const clientX = event.clientX ?? event.touches?.[0]?.clientX;
    const clientY = event.clientY ?? event.touches?.[0]?.clientY;
    if (clientX == null || clientY == null) {
        return null;
    }
    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    const coords = getCanvasCoordinates(e);
    if (!coords) return;
    isDrawing = true;
    
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
    
    // Esconder overlay
    document.querySelector('.canvas-overlay').classList.add('hidden');
}

function draw(e) {
    if (!isDrawing) return;
    
    const coords = getCanvasCoordinates(e);
    if (!coords) return;
    
    if (currentTool === 'pen') {
        ctx.strokeStyle = '#000000';
        ctx.globalCompositeOperation = 'source-over';
    } else {
        ctx.strokeStyle = '#FFFFFF';
        ctx.globalCompositeOperation = 'destination-out';
    }
    
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
    
    updatePreview();
}

function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        ctx.beginPath();
    }
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const eventType = e.type === 'touchstart' ? 'mousedown' :
                     e.type === 'touchmove' ? 'mousemove' : 'mouseup';
    const synthetic = new MouseEvent(eventType, {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(synthetic);
}

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    
    previewCtx.fillStyle = 'white';
    previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    
    // Mostrar overlay novamente
    document.querySelector('.canvas-overlay').classList.remove('hidden');
    
    // Limpar resultado
    document.getElementById('result').classList.add('hidden');
    updateProbabilities([]);
}

function processCanvasPixels() {
    const baseCanvas = document.createElement('canvas');
    baseCanvas.width = GRID_SIZE;
    baseCanvas.height = GRID_SIZE;
    const baseCtx = baseCanvas.getContext('2d');
    baseCtx.drawImage(canvas, 0, 0, GRID_SIZE, GRID_SIZE);

    const baseData = baseCtx.getImageData(0, 0, GRID_SIZE, GRID_SIZE);
    const bounds = computeBoundingBox(baseData);

    let workingCtx = baseCtx;

    if (bounds) {
        const processedCanvas = document.createElement('canvas');
        processedCanvas.width = GRID_SIZE;
        processedCanvas.height = GRID_SIZE;
        const processedCtx = processedCanvas.getContext('2d');
        processedCtx.fillStyle = 'white';
        processedCtx.fillRect(0, 0, GRID_SIZE, GRID_SIZE);

        const boxWidth = bounds.maxX - bounds.minX + 1;
        const boxHeight = bounds.maxY - bounds.minY + 1;
        const targetSize = 20;
        const scale = targetSize / Math.max(boxWidth, boxHeight);
        const destWidth = Math.max(1, Math.round(boxWidth * scale));
        const destHeight = Math.max(1, Math.round(boxHeight * scale));
        const offsetX = Math.round((GRID_SIZE - destWidth) / 2);
        const offsetY = Math.round((GRID_SIZE - destHeight) / 2);

        processedCtx.drawImage(
            baseCanvas,
            bounds.minX,
            bounds.minY,
            boxWidth,
            boxHeight,
            offsetX,
            offsetY,
            destWidth,
            destHeight
        );

        workingCtx = processedCtx;
    }

    const processedData = workingCtx.getImageData(0, 0, GRID_SIZE, GRID_SIZE);
    const pixels = processedData.data;
    const normalizedPixels = new Array(GRID_SIZE * GRID_SIZE);
    let idx = 0;

    for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i];
        const g = pixels[i + 1];
        const b = pixels[i + 2];
        const gray = (r + g + b) / 3;
        const normalized = 1.0 - (gray / 255.0);
        normalizedPixels[idx] = normalized;
        idx++;
    }

    const standardizedPixels = normalizedPixels.map(v => (v - DATA_MEAN) / DATA_STD);

    return {
        normalized: normalizedPixels,
        standardized: standardizedPixels
    };
}

function computeBoundingBox(imageData) {
    const threshold = 0.02;
    let minX = GRID_SIZE;
    let minY = GRID_SIZE;
    let maxX = -1;
    let maxY = -1;
    const data = imageData.data;

    for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
            const index = (y * GRID_SIZE + x) * 4;
            const r = data[index];
            const g = data[index + 1];
            const b = data[index + 2];
            const gray = (r + g + b) / 3;
            const normalized = 1.0 - (gray / 255.0);
            if (normalized > threshold) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    if (maxX === -1 || maxY === -1) {
        return null;
    }

    return { minX, minY, maxX, maxY };
}

// Atualizar preview
function updatePreview() {
    const pixelData = processCanvasPixels();
    const normalizedPixels = pixelData.normalized;
    
    // Desenhar no preview canvas (200x200 para visualização)
    previewCtx.fillStyle = 'white';
    previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    
    const cellSize = previewCanvas.width / GRID_SIZE;
    
    for (let i = 0; i < GRID_SIZE; i++) {
        for (let j = 0; j < GRID_SIZE; j++) {
            const index = i * GRID_SIZE + j;
            const normalized = Math.max(0, Math.min(1, normalizedPixels[index]));
            if (normalized > 0.02) {
                const boosted = Math.min(1, normalized * 1.5);
                const gray = Math.floor(255 * (1 - boosted));
                previewCtx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
            } else {
                previewCtx.fillStyle = '#FFFFFF';
            }
            previewCtx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
        }
    }
}

// Carregar rede neural
async function loadNeuralNetwork() {
    try {
        // Criar estrutura da rede baseada na arquitetura de treino (784 -> 256 -> 128 -> 10)
        const layers = [];
        let previousSize = GRID_SIZE * GRID_SIZE;
        HIDDEN_LAYER_SIZES.forEach(size => {
            layers.push(new Layer(size, previousSize, ActivationType.RELU));
            previousSize = size;
        });
        layers.push(new Layer(OUTPUT_CLASSES, previousSize, ActivationType.LINEAR));
        neuralNetwork = new NeuralNetwork(layers);
        
        // Carregar pesos do ficheiro CSV
        try {
            const response = await fetch('weights/pesos.csv');
            if (!response.ok) {
                throw new Error('Não foi possível carregar os pesos. Certifique-se de que weights/pesos.csv existe.');
            }
            
            const csvText = await response.text();
            neuralNetwork.loadWeightsFromCSV(csvText);
        } catch (weightsError) {
            console.warn('Falha ao carregar pesos:', weightsError);
            throw new Error('Erro ao carregar weights/pesos.csv. Confira se treinou a rede com a arquitetura 784-256-128-10 e está a usar um servidor local.');
        }
        
        console.log('Rede neural carregada com sucesso!');
    } catch (error) {
        console.error('Erro ao carregar rede neural:', error);
        alert('Erro ao carregar a rede neural. Verifique se o ficheiro weights/pesos.csv existe.');
    }
}

// Fazer predição
function predict() {
    if (!neuralNetwork) {
        alert('Rede neural ainda não foi carregada. Aguarde...');
        return;
    }
    
    // Converter canvas para array de pixels
    const pixelData = processCanvasPixels();
    
    // Fazer predição
    const probabilities = neuralNetwork.predict(pixelData.standardized);
    if (!Array.isArray(probabilities) || probabilities.length !== OUTPUT_CLASSES) {
        alert('Os pesos carregados não correspondem à arquitetura atual (10 classes).');
        return;
    }
    
    const { digit, confidence } = getTopPrediction(probabilities);
    displayResult(digit, confidence, probabilities);
}

// Mostrar resultado
function displayResult(prediction, confidence, probabilities) {
    const resultDiv = document.getElementById('result');
    const predictionValue = document.getElementById('predictionValue');
    const predictionLabel = document.getElementById('predictionLabel');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidencePercentage = document.getElementById('confidencePercentage');
    
    // Atualizar valores
    predictionValue.textContent = prediction;
    predictionLabel.textContent = `A rede neural identificou o dígito ${prediction}`;
    const safeConfidence = Math.max(0, Math.min(1, confidence));
    confidenceFill.style.width = `${safeConfidence * 100}%`;
    confidencePercentage.textContent = `${(safeConfidence * 100).toFixed(2)}%`;
    
    // Mostrar resultado
    resultDiv.classList.remove('hidden');
    
    updateProbabilities(probabilities);
    
    // Scroll para o resultado
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function getTopPrediction(probabilities) {
    let bestDigit = 0;
    let bestValue = probabilities[0];
    for (let i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > bestValue) {
            bestValue = probabilities[i];
            bestDigit = i;
        }
    }
    return { digit: bestDigit, confidence: bestValue };
}

function updateProbabilities(probabilities) {
    if (!probabilitiesList) return;
    probabilitiesList.innerHTML = '';
    
    if (!probabilities || probabilities.length === 0) {
        const placeholder = document.createElement('li');
        placeholder.classList.add('placeholder');
        placeholder.textContent = 'Sem previsões ainda.';
        probabilitiesList.appendChild(placeholder);
        return;
    }
    
    const sorted = probabilities
        .map((value, digit) => ({ digit, value }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 5);
    
    sorted.forEach(({ digit, value }) => {
        const li = document.createElement('li');
        li.innerHTML = `<span class="digit">${digit}</span><span class="value">${(value * 100).toFixed(2)}%</span>`;
        probabilitiesList.appendChild(li);
    });
}
