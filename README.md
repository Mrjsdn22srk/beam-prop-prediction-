# beam-prop-prediction-
Predicting laser beam propagation distance using CNNs and LightPipes simulation
# Laser Beam Propagation Prediction using Deep Learning

This repository implements a Machine Learning approach to predict the propagation distance of a Gaussian laser beam based on its 2D intensity profile. By combining **Physical Optics** (using the Angular Spectrum Method) with **Computer Vision**, this project demonstrates how Neural Networks can interpret complex diffraction patterns.

## 📌 Project Overview
In optical physics, a laser beam's intensity distribution changes as it propagates through space due to diffraction. Traditionally, determining the distance $z$ requires complex interferometric setups or precise mechanical measurement. 

This project uses a **Convolutional Neural Network (CNN)** to solve the inverse problem: Given a single intensity snapshot (image) of the beam, can we accurately predict the distance it has traveled?

## 🚀 Features
* **Physical Simulation:** Uses the `LightPipes` optical toolbox to generate realistic laser propagation data.
* **Deep Learning Architecture:** A CNN-based regression model implemented in TensorFlow/Keras.
* **Data Driven:** Automatically generates a synthetic dataset of Gaussian beams at various propagation stages.
* **Visualization:** Includes tools to plot intensity profiles and model loss curves.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Physics Engine:** [LightPipes for Python](https://opticspy.github.io/lightpipes/)
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** NumPy, Scikit-learn
* **Visualization:** Matplotlib

## 📂 Project Structure
```text
├── simulation.py       # Physics-based data generation script
├── model.py            # CNN architecture and model definition
├── main.py             # Training loop and evaluation logic
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
