# 🔬 AI Skin Cancer Classification

### CNN-Based Skin Lesion Image Classification

An educational Deep Learning project that uses a Convolutional Neural Network (CNN) to classify skin lesion images into seven categories using the HAM10000 dataset.

## 📌 Project Overview

This project demonstrates how Deep Learning and Computer Vision can be used for skin lesion image classification.

Users can upload a skin image through a Streamlit web application and receive a predicted class along with class probability scores.

## 📊 Dataset Information

- Dataset: HAM10000
- Total Images: 10,015
- Number of Classes: 7
- Image Size: 96 × 96 pixels

### Classification Classes

| Label | Description |
|---|---|
| akiec | Actinic Keratoses |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevi |
| vasc | Vascular Lesions |

## 🧠 Model

- Model Type: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Input Image Size: 96 × 96 × 3
- Output Classes: 7
- Model Format: `.keras`

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- GitHub
- Kaggle

## 🚀 Run the Project Locally

Clone this repository:

```bash
git clone https://github.com/ahmadrazabrall6/skin-cancer-detection-cnn.git
```

Open the project folder:

```bash
cd skin-cancer-detection-cnn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 📁 Project Files

- `app.py` — Streamlit web application
- `requirements.txt` — Project dependencies
- `skin_cancer_final_model.keras` — Trained CNN model

## ⚠️ Disclaimer

This project is developed for educational and learning purposes only. It is not a medical diagnostic tool. Predictions should not be used to diagnose skin cancer or make medical decisions. Always consult a qualified healthcare professional.

## 👨‍💻 Author

**Ahmad Raza Brall**

Machine Learning & Deep Learning Project

GitHub: [ahmadrazabrall6](https://github.com/ahmadrazabrall6)
