import os


def display_menu():
    """Display main menu and return user choice."""
    print("\nMain Menu:")
    print("1. Train a Perceptron")
    print("2. List and Graph Results")

    choice = input("Choose an option (1 or 2): ")
    return choice


def select_training_method():
    """Let the user select a perceptron training method and return choice."""
    print("\nSelect training method:")
    print("1. Standard Perceptron (Simple)")
    print("2. Gradiant Perceptron (Simple-Based)")
    print("3. Adaline Perceptron (Adaline-Based)")

    choice = input("Enter choice : ")

    if choice in ["1", "2", "3"]:
        return choice
    else:
        print("Invalid choice. Exiting.")
        exit(1)


def display_training_menu():
    """Dynamically display dataset selection menu and return choice."""
    datasets = [f for f in os.listdir('./Training') if f.endswith('.json')]
    print("\nAvailable datasets:")

    for idx, dataset in enumerate(datasets, start=1):
        print(f"{idx}. {dataset}")

    choice = input(f"Select dataset (1-{len(datasets)}): ")
    return choice, datasets


def get_file_path(choice, datasets, directory):
    """Map menu choice to file path and return it."""
    if choice.isdigit() and 1 <= int(choice) <= len(datasets):
        return os.path.join(directory, datasets[int(choice) - 1])
    else:
        print("Invalid choice. Exiting.")
        exit(1)


def display_result_files():
    """List all result files in the 'Result' folder and return the list."""
    result_files = [f for f in os.listdir('./Result') if f.endswith('.csv')]
    print("\nAvailable result files:")

    for idx, result_file in enumerate(result_files, start=1):
        print(f"{idx}. {result_file}")

    return result_files
