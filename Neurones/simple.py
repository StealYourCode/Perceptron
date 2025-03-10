#from perceptron import BasePerceptron
from Neurones.perceptron import BasePerceptron

class SimplePerceptron(BasePerceptron):
    def train(self, inputs, labels, epochs=100):
        """Simple perceptron training with step activation function."""
        if not self.weights:
            self.weights = [0] * len(inputs[0])

        for _ in range(epochs):
            error_count = 0
            for input_vector, label in zip(inputs, labels):
                prediction = self.predict(input_vector)
                error = label - prediction

                if error != 0:
                    error_count += 1
                    self.weights = [self.update_weight(w, error, x) for w, x in zip(self.weights, input_vector)]
                    self.bias += self.learning_rate * error

            if error_count == 0:
                break
