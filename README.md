# Handwritten Digit Recognition Using Neural Networks From Scratch

A beginner-friendly implementation of an Artificial Neural Network (ANN) that recognizes handwritten digits (0–9) using Python, NumPy, and Matplotlib — built completely from scratch without TensorFlow, PyTorch, or other machine learning frameworks.

## About the Project

This project focuses on understanding and implementing the fundamental concepts of neural networks by building a handwritten digit recognition system from the ground up.

The model uses the **MNIST dataset**, which contains 70,000 grayscale images of handwritten digits. Each image is 28 × 28 pixels.

The neural network learns patterns from these images and predicts which digit (0–9) is represented.

Instead of relying on prebuilt machine learning libraries, the mathematical operations behind the neural network are implemented manually using NumPy.

## Project Objectives

- Understand the fundamentals of artificial neural networks.
- Learn how neural network layers, neurons, weights, and biases work.
- Implement activation functions such as ReLU and Softmax.
- Understand forward propagation and loss calculation.
- Implement derivatives and backpropagation.
- Apply gradient descent to optimize model parameters.
- Train and evaluate the neural network using MNIST.
- Visualize predictions and model performance.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Mathematical calculations and array operations |
| Matplotlib | Image visualization and plotting |
| MNIST | Handwritten digit dataset |
| Jupyter Notebook | Development and experimentation |

**Important:** TensorFlow, PyTorch, Keras, and Scikit-learn are not used to build or train the neural network.

## Dataset

The project uses the MNIST handwritten digit dataset.

- Total images: 70,000
- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Number of classes: 10 (digits 0–9)
- Image type: Grayscale

Each image is converted into a 784-element array before being passed to the neural network.

## Neural Network Architecture

The model follows a basic feedforward neural network architecture:

**Input Layer → Hidden Layer → Output Layer**

- Input Layer: 784 neurons (28 × 28 pixels).
- Hidden Layer: Learns patterns and features from the input images.
- Output Layer: 10 neurons representing digits 0–9.

### Activation Functions

- ReLU: Introduces non-linearity in the hidden layer.
- Softmax: Converts output scores into probability distributions.

### Learning Algorithms

- Forward Propagation
- Cross-Entropy Loss
- Backpropagation
- Gradient Descent

## Project Workflow

1. Load the MNIST dataset.
2. Preprocess the images and normalize pixel values.
3. Initialize weights and biases.
4. Perform forward propagation.
5. Apply ReLU and Softmax activation functions.
6. Calculate prediction loss.
7. Calculate gradients using backpropagation.
8. Update weights using gradient descent.
9. Train the neural network.
10. Evaluate predictions using test data.
11. Visualize results using Matplotlib.

## Project Structure

```text
Handwritten-Digit-Recognition/
│
├── dataset/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
│
├── neural_network.py
├── requirements.txt
├── README.md
└── .gitignore
```

*The structure above is a suggested organization. Adjust filenames to match your actual project.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Handwritten-Digit-Recognition.git
```

### 2. Navigate to the Project Folder

```bash
cd Handwritten-Digit-Recognition
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Project

If using a Python script:

```bash
python neural_network.py
```

If using Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and run the cells sequentially.

## Requirements

The project uses the following Python libraries:

```text
numpy
matplotlib
```

Jupyter Notebook can also be installed if you prefer working interactively.

## Results and Visualization

The project aims to demonstrate:

- Handwritten digit predictions.
- Comparison between actual and predicted labels.
- Training loss over epochs.
- Model accuracy.
- Visualization of sample MNIST images.

*Add screenshots of your actual predictions, accuracy, and training graphs here after running the completed model.*

## Learning Outcomes

Through this project, I aim to gain practical experience in:

- Understanding the internal working of neural networks.
- Implementing mathematical operations using NumPy.
- Understanding weights, biases, and activation functions.
- Learning how backpropagation updates neural network parameters.
- Understanding gradient descent and optimization.
- Building a machine learning model without relying on prebuilt ML frameworks.

## Current Development Status

**Status: In Development**

This project is being developed progressively as part of my internship learning journey.

The implementation and testing of neural network components are being completed step by step, from basic mathematical concepts to a fully trained handwritten digit recognition model.

## Future Improvements

- Experiment with different hidden layer sizes.
- Compare different learning rates.
- Improve model accuracy and training efficiency.
- Visualize individual neuron activations.
- Test the model with custom handwritten digit images.
- Develop a simple interface for real-time digit recognition.

## Author

Dheeraj Reddy

Aspiring AI/ML Developer

This project is part of my learning journey in Artificial Intelligence and Machine Learning.

---

⭐ If you find this project interesting, feel free to explore the code and follow its development.
