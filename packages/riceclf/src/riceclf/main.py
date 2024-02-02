import logging
import os

from .data_handler import DataHandler
from .gp_engine import GPEngine

# Configuration parameters
CONFIG = {
    'dataset_path': r'packages/riceclf/src/riceclf/Rice_Cammeo_Osmancik.arff',
    'output_path': r'riceclf_output',
    'population_size': 100,
    'crossover_probability': 0.7,
    'mutation_probability': 0.2,
    'number_of_generations': 30,
    'elitism_size': 1
}

# 'population_size': 700         # A smaller population for quicker testing
# 'crossover_probability': 0.9   # A slightly lower crossover to reduce disruption of individuals
# 'mutation_probability': 0.1    # A higher mutation rate to encourage diversity
# 'number_of_generations': 60    # Fewer generations to reduce computation time
# 'elitism_size': 2              # Minimal elitism to ensure the best solution is carried over


# Configure logging to display info level logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(config):
    """
    The main function to execute the GP algorithm for the rice classification task.

    Args:
        config (dict): Configuration parameters.
    """
    # Initialize the data handler and preprocess the data
    data_handler = DataHandler(config['dataset_path'])
    data_handler.preprocess()

    # Check if the data was successfully preprocessed
    if data_handler.X_train is None or data_handler.y_train is None:
        logging.error("Data preprocessing failed. Exiting program.")
        return

    # Initialize and run the genetic programming engine
    gp_engine = GPEngine(
        data_handler,
        population_size=config['population_size'],
        crossover_probability=config['crossover_probability'],
        mutation_probability=config['mutation_probability'],
        number_of_generations=config['number_of_generations'],
        elitism_size=config['elitism_size']
    )
    population, logbook, hof, test_accuracy = gp_engine.run()

    # Plot the evolution of fitness over generations
    fitness_filename = os.path.join(config['output_path'], 'fitness_over_generations.png')
    gp_engine.plot_fitness(logbook, filename=fitness_filename)

    # Save the best individual to a text file
    best_individual_filename = os.path.join(config['output_path'], 'best_individual.txt')
    gp_engine.save_results(hof, filename=best_individual_filename)

    # Log the test accuracy of the best individual
    logging.info(f"Test Accuracy of the best individual: {test_accuracy:.2f}")


if __name__ == "__main__":
    main(CONFIG)
