# Physical Therapy Activity Classification

This project builds a robust machine learning pipeline to classify physical therapy exercises based on multi-sensor time-series data. The workflow includes preprocessing, time- and frequency-domain feature extraction, feature selection using Random Forest, and evaluation via Leave-One-Subject-Out (LOSO) and 10-fold cross-validation.

---

## 📁 Dataset
The dataset includes sensor data files (`test.txt`, `test2.txt`) for different subjects performing physical therapy exercises. Each sample includes 9 sensor channels:
- Accelerometer: `acc_x`, `acc_y`, `acc_z`
- Gyroscope: `gyr_x`, `gyr_y`, `gyr_z`
- Magnetometer: `mag_x`, `mag_y`, `mag_z`

---

## 🧠 Pipeline Overview

### 1. **Noise Filtering**
- Applied a low-pass Butterworth filter with 5Hz cutoff to remove noise.

### 2. **Normalization**
- Used Z-score normalization across all sensor columns.

### 3. **Segmentation**
- Used 1-second windows (25 samples at 25Hz) with 50% overlap.

### 4. **Feature Extraction**
- **Time-domain**: Mean, STD, RMS, Skewness, Kurtosis, ZCR, Energy, Peak count, etc.
- **Frequency-domain**: FFT energy, entropy, spectral centroid, dominant frequency, PSD, bandwidth.

### 5. **Feature Selection**
- Random Forest used to select top 20 most important features.

### 6. **Model Training**
- Trained 4 classifiers: Random Forest, KNN, SVM, Decision Tree.

### 7. **Evaluation**
- **LOSO Evaluation**: Train on Subject 1, test on Subject 2.
- **10-Fold Cross-Validation**: Using only Subject 1 data.

---

## 📊 Results

### 📌 Leave-One-Subject-Out (Train on S1, Test on S2):
- Random Forest performed the best overall.
- SVM had lower generalization performance than others.

### 📌 10-Fold Cross-Validation:
- All models performed well (above 92% accuracy).
- Random Forest again outperformed others.

---

## 📦 Requirements

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```
---

## ▶️ Run Instructions    
Place 'test.txt' and 'test2.txt' in a 'sample_data/ folder'.  
Open and run the notebook file step-by-step.  
Evaluate model performance using LOSO or cross-validation methods.  

---

## 🔮 Future Work  
Test on more subjects.  
Add real-time activity recognition pipeline.  
Experiment with deep learning models (CNN, LSTM).  

