import csv
import matplotlib.pyplot as plt
import numpy as np


def plot_perceptron_from_csv(csv_path):
    """
    Reads perceptron data from a CSV file and plots:
    - Input points (color-coded by label)
    - Decision boundary line

    Args:
        csv_path (str): Path to the CSV file containing perceptron data.
    """
    inputs = []
    labels = []
    parameters = {}

    # Read CSV data
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        section = 'data'
        for row in reader:
            if not row:
                continue  # Skip empty lines
            if row[0] == 'Parameter':
                section = 'params'
                continue
            if section == 'data' and row[0] != 'Input 1':
                inputs.append([float(row[0]), float(row[1])])
                labels.append(int(row[2]))
            elif section == 'params' and row[0] != 'Parameter':
                parameters[row[0]] = float(row[1])

    # Convert lists to numpy arrays
    inputs = np.array(inputs)
    labels = np.array(labels)

    # Plot input points (red for label 0, blue for label 1)
    for label in [0, 1]:
        subset = inputs[labels == label]
        plt.scatter(
            subset[:, 0], subset[:, 1],
            c='red' if label == 0 else 'blue',
            label=f'Class {label}',
            edgecolors='k'
        )

    # Extract the parameters for the perceptron
    w0 = parameters.get('Weight 0', 0)  # Weight for input 1
    w1 = parameters.get('Weight 1', 0)  # Weight for input 2
    bias = parameters.get('Bias', 0)

    # Check if weights are available
    if w1 != 0:  # If W1 is not zero, we can calculate the decision boundary
        # Calculate slope and intercept for the decision boundary
        slope = -w0 / w1  # slope = -W1/W2
        intercept = -bias / w1  # y-intercept = -b/W2

        # Define x values for the decision boundary line
        x_vals = np.linspace(inputs[:, 0].min() - 1, inputs[:, 0].max() + 1, 100)

        # Calculate corresponding y values using the equation of the line
        y_vals = slope * x_vals + intercept

        # Plot decision boundary
        plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    else:
        # If W1 is zero, the boundary is a vertical line at x = -bias/W0
        plt.axvline(x=-bias / w0, color='k', linestyle='--', label='Decision Boundary')

    # Graph aesthetics
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Perceptron Decision Boundary and Input Points')
    plt.legend()
    plt.grid()
    plt.show()
