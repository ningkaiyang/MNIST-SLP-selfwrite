import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Load MNIST training training_data from pandas DataFrame
train_data = pd.read_csv('training_data/mnist_train_small.csv', header=None)
X_train = train_data.iloc[:, 1:].values / 255.0     # regularlize the x-data that goes from 0-255 to a decimal from 0-1
y_train = train_data.iloc[:, 0].values  # .values converts them to numPy values

# Load MNIST test training_data from pandas DataFrame
test_data = pd.read_csv('training_data/mnist_test.csv', header=None)
X_test = test_data.iloc[:, 1:].values / 255.0    # regularlize the x-data that goes from 0-255 to a decimal from 0-1
y_test = test_data.iloc[:, 0].values  # .values converts them to numPy values

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]
    # np.eye(x) creates a x*x identity matrix, and with y being a vector, will return a matrix that has each value of y selecting a row of the identity matrix and then appending to it
    # for example np.eye(10)[1,5,0,4,3] creates an identity matrix 10x10 and then for each value in the y vector it appends that row of the identity matrix to what is returned.
    # so the end result is that it returns the rows 1, 5, 0, 4, 3.
    # that means that it returns [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] and [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] and etc, correctly one hot encoding the stuff.

def train(X_train, y_train, num_classes, learning_rate, batch_size, num_iterations):
    weights = np.random.randn(X_train.shape[1], num_classes) # X_train.shape[1] is taking the width of the X_train data array (784) in order to initialize a (784, 10) matrix of random 0-1 weights to be used in the model
    losses = [] # These arrays are for the graph
    accuracies = []
    for i in range(num_iterations):
        idx = np.random.randint(0, X_train.shape[0], batch_size) # generate an array of batch_size random integers from the length of all the data to 0, which includes every single entry/the number of training samples.
        X_batch = X_train[idx] # take a group of X_batch from X_train, drawing the feature data from random rows.
        y_batch = y_train[idx] # makes the y_batch, drawing the targets from the same random rows.

        # Compute the dot product between the input and the weights (forward pass)
        z = np.dot(X_batch, weights) # dot product of X_batch (matrix of batch_size x 784) and weights (matrix of 784 x 10) is a matrix of (batch_size x 10) of the predictions of the batches based off the hidden layer.
        # Apply the sigmoid function to obtain the predicted output, where each y_pred[i, j] more or less represents the predicted probability that the i-th sample in the mini-batch belongs to class j.
        y_pred = sigmoid(z)

        # Compute the loss by comparing the predicted output with the true output
        y_true = one_hot_encode(y_batch, num_classes)  # converts the batch targets into a one_hot_encode matrix based on the function we created above
        loss = ((y_pred - y_true) ** 2).mean()  # calculates the element-wise square difference between y_true (0 or 1) and y_pred (some random decimal number) and finds the average of them all, saves to loss

        # Compute the error between the true and predicted output
        error = y_pred - y_true # First compute the error between the true and predicted output by subtracting y_pred from y_true. This gives us a matrix of shape (batch_size, 10) where each element represents the difference between the predicted and true outputs for a particular sample and class.
        # Compute the delta by multiplying the error with the derivative of the sigmoid function
        delta = 2 * error * sigmoid_derivative(y_pred) # dL (change in loss) / dz (change in weightd input) = dL/dy_pred * dy_pred/dz. Since L = (y_pred - y_true) ^2, dL/dy_pred = 2 * (y_pred - y_true) = 2 * error. y_pred = sigmoid(z), so dy_pred/dz = sigmoid_derivative(z). Combine this all and dL/dz which we want for the gradient, is equal to 2 * error * sigmoid_derivative(z).
        # delta is a matrix of dimensions (batch_size, 10), and each element has -2 * error * sigmoid_derivative(z) element-wise operation added in, since all these are matrices of same dimensions.
        # dL/dz basically gives us how much we need to change each of the weighted input z in order to minimize loss.
        gradient = np.dot(X_batch.T, delta) # We know how much we need to change each weighted input by, which means that we can backtrack and find how much we need to change each weight by dot product of the X_batch transposed (784, batch_size) and delta (batch_size, 10) ending up with a (784, 10) matrix of gradients to adjust the (784, 10) matrix of weights
        # This moves towards minimizing the loss on the mini-batch, this operation can be understood as summing over all samples in the mini-batch to compute the gradient for each weight.

        # Update the weights using the gradient matrix we calculated, multiplied by learning rate to slow it down and make steps smaller. Remember it goes the opposite direction of the slopes we have calclated so it is a subtraction.
        weights -= learning_rate * gradient

        # Print the loss every N iterations
        if i % 1000 == 0:
            print(f'Iteration {i}: Loss {loss}')
            accuracy = test(X_test, y_test, weights)
            print(f'Iteration {i}: Accuracy {accuracy}')
            losses.append(loss)
            accuracies.append(accuracy)

    # Plot the graphs
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(accuracies)
    plt.title('Accuracy')

    plt.show()

    # Save the trained weights
    save_weights(weights)

    # Return the trained weights when done, so the final test function can run
    return weights

# Define the test function
def test(X_test, y_test, weights):
    # Compute the dot product between the input and the weights (forward pass)
    z = np.dot(X_test, weights) # z is a (10000, 10) matrix of the input 10000 test samples and their classes predicted (pre-sigmoid)
    y_pred = sigmoid(z) # Apply the sigmoid function to obtain the predicted output
    # Compute the predicted class by taking the argmax of y_pred along axis 1 of the (10000, 10) matrix
    predictions = np.argmax(y_pred, axis=1) # Axis = 1 collapses the columns and returns the index of the max element of each row
    # Compute the accuracy by comparing predictions with true labels - element-wise comparison with True/False for each and average the equal comparisons
    accuracy = (predictions == y_test).mean()
    return accuracy

def save_weights(weights):
    # Create the directory if it doesn't exist
    os.makedirs('previous_generated_weights', exist_ok=True)
    
    # Generate a unique filename based on current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'previous_generated_weights/weights_{timestamp}.npy'
    
    # Save the weights
    np.save(filename, weights)
    print(f"Weights saved to {filename}")

num_classes = 10
learning_rate = 1
batch_size = 10
num_iterations = 100000

weights = train(X_train, y_train, num_classes, learning_rate, batch_size, num_iterations)

# Test model on test data and compute accuracy.
accuracy = test(X_test, y_test, weights)
print(f'Test accuracy: {accuracy}')

def test_with_saved_weights(weights_path, dataset_path):
    # Load the weights
    loaded_weights = np.load(weights_path)
    
    # Load the dataset
    data = pd.read_csv(dataset_path, header=None)
    X = data.iloc[:, 1:].values / 255.0
    y = data.iloc[:, 0].values
    
    # Run the test
    accuracy = test(X, y, loaded_weights)
    print(f'Test accuracy using weights from {weights_path} on dataset {dataset_path}: {accuracy}')

# Example usage (commented out):
# test_with_saved_weights('previous_generated_weights/weights_20250228_164241.npy', 'training_data/mnist_test.csv')