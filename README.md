# Tifinagh Character Recognition using RGB Images and MLP (from scratch)

This project implements a **Multi-Layer Perceptron (MLP)** in pure NumPy to classify **Tifinagh characters** from RGB images. It includes:

- Manual implementation of a neural network
- Preprocessing (resizing, normalization, encoding)
- Training with **Adam**, **L2 regularization**, and **He initialization**
- Visualization of training curves, predictions, and weight histograms
- Confusion matrix for final evaluation

---

## Dataset

- **Name:** AMHCD-64
- **Content:** 64x64 RGB images of Tifinagh characters
- **Classes:** 33 different Tifinagh characters
- **Preprocessing:**
  - Resized to 32×32
  - Normalized (mean 0, std 1)
  - One-hot encoding of labels

---

##  Model Architecture

| Layer            | Details               |
|------------------|------------------------|
| Input Layer      | 3072 (32x32x3) neurons |
| Hidden Layer 1   | 64 neurons, ReLU       |
| Hidden Layer 2   | 32 neurons, ReLU       |
| Output Layer     | 33 neurons, Softmax    |

Other features:
- **He Initialization**
- **L2 Regularization**
- **Adam Optimizer**

---

##  Training Details

- **Epochs:** 100  
- **Batch size:** 64  
- **Initial learning rate:** 0.001  
- **Decay:** ×0.7 every 20 epochs  
- **Training / Validation split:** 80 / 20

---

## 📊 Results

- **Best validation accuracy:** ~95.5%
- **Visualizations:**
  - `training_curves.png`
  - `confusion_matrix.png`
  - `weights_hist_epoch_XX.png`
  - `predictions_epoch_XX.png`

---

## 📁 Structure

project/
├── data/ # Contains AMHCD_64 dataset
├── tifinagh_rgb_classifier.py
├── training_curves.png
├── confusion_matrix.png
├── predictions_epoch_.png
├── weights_hist_epoch_.png
├── model.pkl # Saved model 
└── README.md


---

## 🔧 Usage

```bash
pip install numpy matplotlib opencv-python scikit-learn joblib
python tifinagh_rgb_classifier.py

