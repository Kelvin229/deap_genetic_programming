import logging

from data_handler import DataHandler
from gp_engine import GPEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # Initialize the data handler with the path to your dataset
    data_handler = DataHandler(r'C:\Users\Path\to\A1\partB\Rice_Cammeo_Osmancik.arff')

    # Load and preprocess the data
    data_handler.preprocess()

    # Check if data was successfully preprocessed
    if data_handler.X_train is None or data_handler.y_train is None:
        logging.error("Data preprocessing failed. Exiting program.")
        return  # Exit the program if data preprocessing failed

    # Initialize the GP engine with the preprocessed data
    gp_engine = GPEngine(data_handler)

    # Run the genetic programming process
    population, logbook, hof, test_accuracy = gp_engine.run()

    # Plot and save the fitness evolution
    gp_engine.plot_fitness(logbook,
                           filename=r"C:\Users\path\to\A1\partB\output\fitness_over_generations.png")

    # Saving the best individual to a file
    gp_engine.save_results(hof, filename="C:\\Users\\path\\to\\A1\\partB\\output\\best_individual.txt")

    # Log test accuracy
    logging.info(f"Test Accuracy of the best individual: {test_accuracy:.2f}")


if __name__ == "__main__":
    main()
