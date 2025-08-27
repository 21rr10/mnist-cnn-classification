# MNIST Handwritten Digit Classification Using CNN

---

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. The CNN architecture includes Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers following industrial standards.

**Key highlights:**
- Achieves over 95% accuracy on the MNIST test set (typically ~98.5%)
- Implements best practices including batch normalization and callbacks
- Visualizes training history and sample prediction results

---

## Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Required Libraries

Install the necessary packages via pip:
pip install tensorflow matplotlib seaborn numpy scikit-learn

---

## Usage

1. Clone this repository:
git clone https://github.com/21rr10/mnist-cnn-classification.git
cd mnist-cnn-classification


2. Open the Jupyter notebook file:

jupyter notebook mnist_cnn_implementation.ipynb


3. Run all cells sequentially to:
- Load and preprocess the MNIST dataset
- Build, compile, and train the CNN model
- Visualize training metrics and sample predictions
- Save the trained model (`mnist_cnn_model.keras`)

---

## Project Structure

├── mnist_cnn_implementation.ipynb # Jupyter notebook with code and visualizations
├── mnist_cnn_model.keras # Saved Keras model file (optional)
├── README.md # This README file
├── requirements.txt # List of required Python packages (optional)
└── .gitignore # Git ignore file

---

## Results

- Test accuracy typically exceeds 95% within 10-15 minutes of training
- Training and validation accuracy/loss curves
- Confusion matrix analysis for digit-wise performance
- Sample predictions display with confidence levels

---

## Future Work

- Add data augmentation to improve generalization
- Experiment with deeper architectures or transfer learning
- Build a web app with Flask/FastAPI for real-time digit recognition
- Containerize the project using Docker for easy deployment

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- MNIST dataset by Yann LeCun
- TensorFlow and Keras for deep learning frameworks
- Open-source libraries used: NumPy, Matplotlib, Seaborn, scikit-learn
