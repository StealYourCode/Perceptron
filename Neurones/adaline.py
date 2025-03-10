from Neurones.perceptron import BasePerceptron

class AdalinePerceptron(BasePerceptron):
    def train(self, inputs, labels, epochs=100):
        """Adaline (Adaptive Linear Neuron) training using gradient descent."""
        if not self.weights:
            self.weights = [0] * len(inputs[0])

        for _ in range(epochs):
            for input_vector, label in zip(inputs, labels):
                weighted_sum = sum(w * x for w, x in zip(self.weights, input_vector)) + self.bias
                error = label - weighted_sum  # Uses continuous error instead of step function

                self.weights = [w + self.learning_rate * error * x for w, x in zip(self.weights, input_vector)]
                self.bias += self.learning_rate * error
