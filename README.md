# 🫁 AI-Powered Chest X-Ray Pneumonia Detection using ResNet18

An AI-based deep learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a ResNet18 convolutional neural network.

## 📌 Project Overview

This project uses a pretrained ResNet18 model and transfer learning to analyze chest X-ray images.

The system performs two-class classification:

- NORMAL
- PNEUMONIA

The trained model is integrated into a Streamlit web application where users can upload a chest X-ray image and receive a prediction with class probabilities.

## ✨ Features

- Chest X-ray image upload
- Image preprocessing and normalization
- ResNet18 deep learning model
- NORMAL/PNEUMONIA classification
- Prediction confidence
- Class probability display
- Streamlit web interface
- CPU-based inference

## 🛠️ Technologies Used

- Python
- PyTorch
- TorchVision
- ResNet18
- OpenCV
- PIL
- Streamlit
- NumPy

## 📊 Dataset

The project uses the **Chest X-Ray Pneumonia** dataset containing NORMAL and PNEUMONIA chest X-ray images.

Dataset structure:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
