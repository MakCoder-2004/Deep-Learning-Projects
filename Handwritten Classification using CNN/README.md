# Handwritten Digit Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model achieves high accuracy in recognizing and classifying digits from 0 to 9.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project demonstrates the implementation of a CNN using TensorFlow and Keras to classify handwritten digits. The model is trained on the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits (0-9).

## Technologies Used
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

## Model Architecture
The CNN architecture consists of the following layers:

1. **Convolutional Layer 1**: 32 filters, (3x3) kernel, ReLU activation
2. **Max Pooling Layer 1**: (2x2) pool size
3. **Convolutional Layer 2**: 64 filters, (3x3) kernel, ReLU activation
4. **Max Pooling Layer 2**: (2x2) pool size
5. **Convolutional Layer 3**: 64 filters, (3x3) kernel, ReLU activation
6. **Max Pooling Layer 3**: (2x2) pool size
7. **Flatten Layer**: Converts 3D feature maps to 1D feature vector
8. **Dense Layer 1**: 128 neurons, ReLU activation
9. **Output Layer**: 10 neurons (for digits 0-9), softmax activation

## Dataset
The model uses the MNIST dataset, which consists of 70,000 grayscale images (28x28 pixels) of handwritten digits (60,000 for training and 10,000 for testing).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/handwritten-digit-classification.git
cd handwritten-digit-classification
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Open and run the `main.ipynb` Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

2. The notebook includes the following sections:
   - Loading and preprocessing the MNIST dataset
   - Building the CNN model
   - Training the model
   - Evaluating the model on test data
   - Making predictions

## Results
The model achieves the following performance metrics:
- Training Accuracy: ~99.5%
- Test Accuracy: ~98.7%

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
