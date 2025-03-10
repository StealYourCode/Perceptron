from Neurones.perceptron import BasePerceptron
import numpy as np

class GradientPerceptron(BasePerceptron):
    def train(self, inputs, labels, epochs=100):
        """
        Train perceptron using gradient descent.
        Unlike SimplePerceptron, this updates weights continuously using a gradient-based approach.
        """
        if not self.weights:
            self.weights = np.zeros(len(inputs[0]))

        inputs = np.array(inputs)
        labels = np.array(labels)

        for _ in range(epochs):
            errors = []

            for input_vector, label in zip(inputs, labels):
                prediction = self.predict(input_vector)
                error = label - prediction
                errors.append(error)

                # Update weights using gradient-based rule
                self.weights += self.learning_rate * error * input_vector
                self.bias += self.learning_rate * error  # Bias update

            # Stop early if all errors are 0
            if all(e == 0 for e in errors):
                break

    def predict(self, input_vector):
        """
        Predict using the step activation function.
        Returns 1 if the weighted sum >= 0, otherwise returns 0.
        """
        weighted_sum = np.dot(input_vector, self.weights) + self.bias
        return 1 if weighted_sum >= 0 else 0
