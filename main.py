import json
import os
from DataAnalyse.graph import plot_perceptron_from_csv
from Interface.menu import (
    display_menu,
    select_training_method,
    display_training_menu,
    get_file_path,
    display_result_files
)
from Neurones.adaline import AdalinePerceptron
from Neurones.simple import SimplePerceptron
from Neurones.gradient import GradientPerceptron  # Ensure this exists


def load_data(json_file_path):
    """Load training data from JSON file"""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data['inputs'], data['labels']


def handle_graph_generation():
    """Allow user to select a result file and generate a graph"""
    result_files = display_result_files()

    if not result_files:
        print("No result files available.")
        return

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
            # Select training method
            training_method = select_training_method()

            # Select dataset
            choice, datasets = display_training_menu()
            json_file_path = get_file_path(choice, datasets, './Training')

            # Load training data
            training_inputs, training_labels = load_data(json_file_path)

            # Choose correct perceptron type
            if training_method == '1':
                perceptron = SimplePerceptron(learning_rate=0.1, bias=0)
            elif training_method == '2':
                perceptron = GradientPerceptron(learning_rate=0.1, bias=0)
            elif training_method == '3':
                perceptron = AdalinePerceptron(learning_rate=0.1, bias=0)
                pass
            else:
                print("Invalid perceptron type selected. Exiting.")
                exit(1)

            # Train perceptron
            perceptron.train(training_inputs, training_labels)

            # Get corresponding testing file path
            testing_file_path = json_file_path.replace('./Training', './Testing')

            # Validate testing file exists
            if not os.path.exists(testing_file_path):
                print(f"Error: Testing file not found at {testing_file_path}")
                continue

            # Load testing data
            testing_inputs, testing_labels = load_data(testing_file_path)

            # Test perceptron
            print("\nTesting results:")
            for input_data, label in zip(testing_inputs, testing_labels):
                prediction = perceptron.predict(input_data)
                print(f"Input: {input_data} -> Predicted: {prediction} (Expected: {label})")

            # Save results
            save_choice = input("\nSave results? (y/n): ").lower()
            if save_choice == 'y':
                save_path = input("Enter save path for perceptron graph data: ")
                if not save_path:
                    print("No save path provided. Exiting.")
                    exit(1)
                perceptron.save_info(save_path, training_inputs, training_labels)
                print(f"Perceptron graph data saved to {save_path}")

        elif choice == '2':
            # Generate graph from results
            handle_graph_generation()

        else:
            print("Invalid option. Please select either 1 or 2.")
            continue

        # Ask if user wants to continue
        continue_choice = input("\nDo you want to perform another task? (y/n): ").lower()
        if continue_choice != 'y':
            print("Exiting program.")
            break


if __name__ == "__main__":
    main()
