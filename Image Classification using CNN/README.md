# 🚀 Image Classification with CNN (CIFAR-10)

A complete deep-learning workflow: **train a CNN on CIFAR-10**, save the best model, and deploy a **Streamlit web app** for real-time image classification. This project is ideal for learning convolutional neural networks, model training pipelines, and lightweight app deployment.

---

## 📁 Project Contents

| File                      | Description                                                                      |
| ------------------------- | -------------------------------------------------------------------------------- |
| **`main.ipynb`**          | Notebook that loads CIFAR-10, builds & trains the CNN, and saves the best model. |
| **`best_model.h5`**       | Best model saved via `ModelCheckpoint`.                                          |
| **`image_classifier.h5`** | Optional additional model.                                                       |
| **`app.py`**              | Streamlit app for image upload, preprocessing, and prediction.                   |
| **`requirements.txt`**    | Dependencies for training and running the app.                                   |

---

## 🧠 Project Overview

This CNN classifies CIFAR-10 images into:

**Plane · Car · Bird · Cat · Deer · Dog · Frog · Horse · Ship · Truck**

The project includes both **model training** and a **web app** for real-time inference.

---

## 📊 Dataset & Preprocessing

* **Dataset:** CIFAR-10 (60,000 images, 32×32 RGB)
* **Normalization:** pixel values / 255.0
* **Augmentation:** rotation, shifting, flipping, zooming

These steps improve generalization and reduce overfitting.

---

## 🏗️ Model Architecture

### 🔹 Feature Extraction

```
Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout
Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout
Conv2D(128) → BN → MaxPool → Dropout
```

### 🔹 Classifier Head

```
Flatten → Dense(256, relu) → BN → Dropout(0.5) → Dense(10, softmax)
```

### 🔧 Training

* Optimizer: Adam (lr=1e-3)
* Loss: sparse categorical crossentropy
* Batch size: 64
* Epochs: up to 50
* Callbacks: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

---

## 🖥️ Streamlit App

The app lets users upload images and instantly get predictions.

### Preprocessing inside the app:

1. Convert to RGB
2. Resize to **32×32**
3. Normalize to **[0,1]**
4. Predict with shape (1, 32, 32, 3)

The UI shows:

* Original uploaded image
* Preprocessed image
* Prediction result
* Probability bar chart

---

## ▶️ How to Run the App

### 1️⃣ Install dependencies

```powershell
pip install -r "requirements.txt"
```

### 2️⃣ Run Streamlit

```powershell
streamlit run app.py"
```

---

## 🛠️ Troubleshooting

* Missing streams → reinstall via `py -m pip install streamlit`
* Missing model → ensure `best_model.h5` exists
* Version issues → pin TensorFlow, Numpy, Pillow versions

---

## 🔄 Reproducibility

* Use the same environment and package versions
* Rerun `main.ipynb`

---

## 🙌 Acknowledgements

Built with:

* TensorFlow / Keras
* Streamlit
* CIFAR-10 Dataset
