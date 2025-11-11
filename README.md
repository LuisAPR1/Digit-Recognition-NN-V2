# MNIST Digit Recognition – Java Neural Network

End-to-end digit recognizer built from scratch: a Java backend that trains a feed-forward neural network on MNIST, and a responsive web frontend that lets you sketch digits and see predictions instantly.

## Contents

- [Overview](#overview)  
- [Network Architecture](#network-architecture)  
- [Repository Layout](#repository-layout)  
- [How It Works](#how-it-works)  
- [Requirements](#requirements)  
- [Getting Started](#getting-started)  
- [Technical Details](#technical-details)  
- [Troubleshooting](#troubleshooting)

---

## Overview

- **Backend (Java):** trains the neural network via mini-batch backpropagation, exports weights to CSV, and logs loss evolution.  
- **Frontend (HTML/CSS/JS):** modern UI that mirrors the MNIST preprocessing pipeline: draw, center, scale to 28×28, normalize, and classify in the browser.

Key features:
- Pure Java implementation (no ML frameworks)
- ReLU hidden layers + Softmax output
- Mini-batch SGD with data standardization
- Live preview of the 28×28 input and top-5 predictions

---

## Network Architecture

```
Input  : 784 neurons (flattened 28×28)
Hidden : 256 neurons, ReLU
Hidden : 128 neurons, ReLU
Output : 10 logits + Softmax
```

- **Learning rate:** 0.1 (configurable)  
- **Batch size:** 64 (mini-batch SGD)  
- **Loss:** Cross-Entropy  
- **Early stop:** training halts when average loss < 0.05 or if loss increases for 10 epochs

---

## Repository Layout

```
├── data/                # MNIST IDX files (train/test)
├── scripts/             # Helper scripts
│   ├── download_mnist.py
│   ├── compile.bat
│   ├── open.bat
│   └── server.py
├── src/java/            # Java sources (training)
│   ├── Main.java
│   ├── MnistLoader.java
│   ├── NeuralNetwork.java
│   ├── Layer.java
│   ├── Neuron.java
│   └── ActivationType.java
├── web/                 # Frontend
│   ├── index.html
│   ├── app.js
│   ├── neural-network.js
│   └── style.css
├── weights/             # Trained weights + loss history
│   ├── pesos.csv
│   └── mse_values.txt
└── README.md
```

---

## How It Works

1. **Training (Java)**
   - `MnistLoader` streams IDX files, normalizing each pixel to `(value/255 - 0.1307) / 0.3081` (same stats as PyTorch).  
   - `NeuralNetwork` performs forward/backward passes with ReLU hidden layers and Softmax outputs.  
   - Gradients are accumulated per batch and applied after each batch.  
   - Weights are saved to `weights/pesos.csv`; loss per epoch goes to `weights/mse_values.txt`.

2. **Inference (Web)**
   - Sketch digits on a 400×400 canvas; the UI centers and scales the drawing to 28×28, normalizes it, and mirrors the Java preprocessing.  
   - `web/neural-network.js` loads `weights/pesos.csv`, recreates the same architecture, and runs Softmax to generate probabilities.  
   - The dashboard shows the predicted digit, confidence, top-5 classes, and the exact 28×28 preview passed to the model.

---

## Requirements

| Purpose | Tools |
|---------|-------|
| Training | Java JDK 8+ |
| Data download / local server | Python 3.x |
| Frontend | Modern browser (Chrome, Edge, Firefox, Safari) |

> **Tip:** keep the IDX files (~60 MB) inside `data/` but out of version control (`.gitignore` already handles this).

---

## Getting Started

### 1. Download MNIST (first run)
```bash
python scripts/download_mnist.py
```

### 2. Compile the Java sources
```bash
# Windows (batch script)
scripts\compile.bat

# Manual (any OS, run from repo root)
javac src\java\*.java
```

### 3. Train and test
```bash
java -cp src\java Main
```
This trains the network, writes `weights/pesos.csv`, logs loss per epoch, and prints test accuracy.  
To skip training and reuse existing weights:
```bash
java -cp src\java Main --test-only
```

### 4. Launch the web app
```bash
# Option A: helper script (Windows)
scripts\open.bat

# Option B: Python HTTP server
python scripts/server.py
# or
python -m http.server 8000
```
Visit `http://localhost:8000/web/index.html`, draw a digit, and click “Run inference”.

---

## Technical Details

### Preprocessing
1. Capture from 400×400 canvas  
2. Downscale to 28×28 and center using the bounding box  
3. Convert to grayscale, invert (`1 - gray/255`)  
4. Standardize with MNIST stats (`mean=0.1307`, `std=0.3081`)  
5. Flatten to 784-value vector

### Backpropagation
1. Forward pass stores inputs/outputs per layer  
2. Cross-Entropy loss with Softmax combined gradient (`softmax - target`)  
3. ReLU derivative used for hidden layers  
4. Gradients accumulated over batch, then weights/biases updated

### File Formats
- **IDX**: original MNIST files stored in `data/`  
- **weights/pesos.csv**: each line = neuron weights + bias for the trained model  
- **weights/mse_values.txt**: per-epoch loss log

---

## Troubleshooting

| Problem | Cause / Fix |
|---------|-------------|
| `FileNotFoundException` when training | Ensure `data/` contains the four IDX files and run `java -cp src\java Main` from the repo root (so `data/` and `weights/` resolve correctly). |
| Web app cannot load `weights/pesos.csv` | Serve via HTTP (not `file://`) and verify the file exists under `/weights`. |
| Python script not found | Install Python 3 and make sure it’s on PATH. Alternatively run any HTTP server pointing to the repo root. |
| Accuracy plateaus ~96% | Check that inputs are centered, contrasty, and similar to MNIST digits; consider retraining or tweaking `lossThreshold` / `learningRate`. |

---

## License & Notes
- Educational project exploring neural networks without external ML libraries.  
- MNIST dataset courtesy of Yann LeCun and collaborators.  
- Feel free to experiment: change the architecture, add convolutional layers, or port the weights to TF.js.
