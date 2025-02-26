import json
import os
from Neurones.perceptron import Perceptron
from DataAnalyse.graph import plot_perceptron_from_csv
from Interface.menu import display_menu, display_training_menu, get_file_path, display_result_files


def load_data(json_file_path):
    """Load training data from JSON file"""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data['inputs'], data['labels']

def handle_graph_generation():
    """Allow user to select a result file and generate a graph"""
    result_files = display_result_files()
    choice = input(f"Select result file (1-{len(result_files)}): ")
    if choice.isdigit() and 1 <= int(choice) <= len(result_files):
        result_file = result_files[int(choice) - 1]
        file_path = os.path.join('./Result', result_file)
        plot_perceptron_from_csv(file_path)
        print(f"Graph generated for {result_file}")
    else:
        print("Invalid choice. Exiting.")

def main():
    while True:
        # Display main menu
        choice = display_menu()

        if choice == '1':
            # Let user choose dataset for perceptron training
            choice, datasets = display_training_menu()
            json_file_path = get_file_path(choice, datasets, './Training')

            # Load selected training data
            training_inputs, training_labels = load_data(json_file_path)

            # Create and train perceptron
            perceptron = Perceptron(learning_rate=0.1, bias=0)
            perceptron.train(training_inputs, training_labels)

            # Get corresponding testing data file path
            testing_file_path = json_file_path.replace('./Training', './Testing')

            # Check if the corresponding testing file exists
            if not os.path.exists(testing_file_path):
                print(f"Error: Testing file not found at {testing_file_path}")
                continue

            # Load selected testing data
            testing_inputs, testing_labels = load_data(testing_file_path)

            # Test perceptron
            print("\nTesting results:")
            for i, (input_data, labels) in enumerate(zip(testing_inputs, testing_labels)):
                prediction = perceptron.predict(input_data)
                print(f"Input: {input_data} -> Predicted: {prediction} (Expected: {labels})")

            # Save results
            save_choice = input("\nSave results? (y/n): ").lower()
            if save_choice == 'y':
                save_path = input("Enter save path for perceptron graph data: ")
                if save_path == "":
                    print("No save path provided. Exiting.")
                    exit(1)
                perceptron.save_info(save_path, training_inputs, training_labels)
                print(f"Perceptron graph data saved to {save_path}")

        elif choice == '2':
            # Generate graph from result files
            handle_graph_generation()

        else:
            print("Invalid option. Please select either 1 or 2.")
            continue

        # Ask user if they want to perform another task
        continue_choice = input("\nDo you want to perform another task? (y/n): ").lower()
        if continue_choice != 'y':
            print("Exiting program.")
            break


if __name__ == "__main__":
    main()
