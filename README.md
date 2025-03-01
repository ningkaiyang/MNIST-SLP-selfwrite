# MNIST Single Layer Perceptron

## Description

This project, created on July 21, 2023 while I was attending COSMOS (California State Summer School For Mathematics and Science) at UC Davis to study Machine Learning, is an implementation of a neural network machine learning model from scratch. It specifically focuses on creating a Single Layer Perceptron (SLP) for the MNIST handwritten digits dataset using only NumPy and Pandas for data manipulation. The core of this project lies in the matrix-based mathematical derivation of the SLP. I wished to challenge myself and uncover a little bit of the "black box" of machine learning by taking a fundamental approach to neural network construction without relying on high-level machine learning libraries.

## Features

- Implementation of a Single Layer Perceptron using NumPy
- Training on the MNIST dataset for handwritten digit recognition
- Visualization of training loss and accuracy
- Saving and loading of trained model weights
- Testing functionality with saved weights

## Dependencies

- Python 3.13
- NumPy
- Pandas
- Matplotlib

## Usage

1. Ensure all dependencies are installed.
2. Ensure the MNIST dataset files (`mnist_train_small.csv` and `mnist_test.csv`) are in the `training_data` directory.
3. Run the main script:
   ```
   python main.py
   ```
4. The script will train the model, save the weights, and display performance graphs.
5. To test with saved weights, modify the `test_with_saved_weights` function call in `main.py` with the desired weights file and dataset.

## File Structure

```
.
├── previous_generated_weights/
│   └── [Saved model weights]
├── training_data/
│   ├── mnist_test.csv
│   └── mnist_train_small.csv
├── main.py
└── README.md
```

## Author

Nickolas Yang

## Date

Created: July 21, 2023

GitHub Port of Project (random inspiration to document it over from Google Colab): Feb 28, 2025