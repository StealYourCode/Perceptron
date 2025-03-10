import csv

class BasePerceptron:
    def __init__(self, learning_rate=0.1, bias=0):
        self.learning_rate = learning_rate
        self.bias = bias
        self.weights = []

    def update_weight(self, weight, error, input_value):
        """Update weight based on error and input value."""
        return weight + self.learning_rate * error * input_value

    def predict(self, input_vector):
        """Make a prediction based on the weighted sum."""
        weighted_sum = sum(w * x for w, x in zip(self.weights, input_vector)) + self.bias
        return 1 if weighted_sum >= 0 else 0

    def save_info(self, filename, inputs, labels):
        """Save perceptron parameters and training data."""
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
