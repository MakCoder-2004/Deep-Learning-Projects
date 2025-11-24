# Potato Disease Classification Using CNN

Project to classify potato leaf images into disease categories (e.g. Early blight, Late blight, Healthy) using a Convolutional Neural Network (CNN).

## Overview
This repository contains code, a trained model, and a Jupyter notebook used to train and evaluate a CNN for potato disease classification using images from the PlantVillage dataset (potato subset). The goal is to provide an easy-to-run notebook for exploration, training, and quick inference with a pre-trained model.

## Project structure
- `model/` - contains the trained model file:
  - `potatoes.h5` - pre-trained Keras model (TensorFlow backend)
- `Notebook/`
  - `Main.ipynb` - main notebook with data loading, preprocessing, model definition, training, evaluation, and inference examples
  - `PlantVillage Dataset/` - dataset folder (images by class)

> Note: If the dataset folder contains the full PlantVillage potato images, ensure image counts and class structure match the notebook's assumptions before re-training.

## Requirements
Recommended: Python 3.8+ and the following packages.

Typical minimal set (can be installed via pip):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate; pip install --upgrade pip
pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python pillow jupyter
```

If you prefer, create a `requirements.txt` with the pinned versions you used and run `pip install -r requirements.txt`.

## How to run
1. Create and activate a virtual environment and install dependencies (see commands above).
2. Start Jupyter and open `Notebook/Main.ipynb`:

```powershell
jupyter notebook Notebook\Main.ipynb
```

3. Run cells top-to-bottom. The notebook includes sections for:
   - Data loading & preprocessing
   - Data augmentation (if used)
   - Model architecture definition
   - Training & checkpoints
   - Evaluation & visualization
   - Inference / sample predictions

## Quick inference example (Python)
If you want to load the saved model and run a quick prediction from a script or REPL, an example:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# adjust path as needed
MODEL_PATH = 'model/potatoes.h5'
img_path = 'Notebook/PlantVillage Dataset/Potato___Early_blight/001187a0-...JPG'

model = load_model(MODEL_PATH)

img = image.load_img(img_path, target_size=(224, 224))  # use the input size expected by the model
x = image.img_to_array(img)
x = x / 255.0
x = np.expand_dims(x, axis=0)

pred = model.predict(x)
# If model uses softmax over classes:
pred_class = np.argmax(pred, axis=1)[0]

print('Raw model output:', pred)
print('Predicted class index:', pred_class)
```

Adjust `target_size` and preprocessing to match what the notebook/training code used.

## Training notes
- The notebook demonstrates how to build and train the CNN. Typical steps include:
  - Resizing and normalizing images
  - Balancing or augmenting classes (if dataset is imbalanced)
  - Using `ImageDataGenerator` or tf.data pipelines for efficient input
  - Checkpointing best model weights (ModelCheckpoint)
  - Early stopping to avoid overfitting
- For reproducibility, pin random seeds and record package versions.

## Evaluation
- The notebook contains code to compute accuracy, confusion matrix, and per-class precision/recall. Use those cells to evaluate model performance on held-out test data.

## Dataset & citation
This project uses images from the PlantVillage dataset (public dataset of plant leaf images). If you publish results or use this model in a paper, please cite the PlantVillage dataset and adhere to its license.

## Model file location
The model file (if present) is at:
```
model/potatoes.h5
```
Load it via `tensorflow.keras.models.load_model` as shown above.

## Next steps and improvements
- Improve model accuracy by trying deeper CNN backbones (e.g., MobileNet, EfficientNet) with transfer learning
- Add a small web demo (Flask / FastAPI) for uploading an image and returning predictions
- Create a small unit test suite for data preprocessing functions
- Add a `requirements.txt` with pinned package versions for reproducibility

## Known issues & troubleshooting
- If you encounter GPU/driver issues, try running on CPU (set CUDA_VISIBLE_DEVICES) or match TensorFlow version to your CUDA toolkit
- If images are not found, verify the dataset path and filenames; the notebook assumes a certain folder layout (one folder per class)
