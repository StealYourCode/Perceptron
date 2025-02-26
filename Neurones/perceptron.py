import csv

class Perceptron:
    def __init__(self, learning_rate, bias):
        self.learning_rate = learning_rate
        self.bias = bias
        self.weights = []

    def update_weight(self, weight, error, input_value):
        return weight + self.learning_rate * error * input_value

    def train(self, inputs, labels, epochs=100):
        if not self.weights:
            self.weights = [0] * len(inputs[0])

        for empty in range(epochs):
            error_count = 0
            for entries_number in range(len(inputs)):
                input_vector = inputs[entries_number]
                label = labels[entries_number]

                prediction = self.predict(input_vector)
                error = (label - prediction)

                if error != 0:
                    error_count += 1
                    for j in range(len(self.weights)):
                        self.weights[j] = self.update_weight(self.weights[j], error, input_vector[j])
                    self.bias += self.learning_rate * error * 1
            if error_count == 0:
                break

    def predict(self, input_vector):
        weighted_sum = 0
        for i in range(len(self.weights)):
            weighted_sum += self.weights[i] * input_vector[i]
        weighted_sum += self.bias

        if weighted_sum >= 0:
            return 1
        else:
            return 0

    def save_info(self, filename, inputs, labels):
        """
        Save perceptron parameters and training data required for graph visualization.
        Includes input points with labels and perceptron weights/bias for the decision boundary.
        """
        filename = f"./Result/{filename}.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(['Input 1', 'Input 2', 'Label'])
            for inp, label in zip(inputs, labels):
                writer.writerow([inp[0], inp[1], label])

            writer.writerow([])
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['Learning Rate', self.learning_rate])
            writer.writerow(['Bias', self.bias])
            for i, w in enumerate(self.weights):
                writer.writerow([f'Weight {i}', w])
