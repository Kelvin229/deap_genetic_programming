import argparse
import logging
import os
import random

from .data_handler import DataHandler
from .gp_engine import GPEngine

# Configuration parameters
CONFIG = {
    "dataset_path": r"packages/riceclf/src/riceclf/Rice_Cammeo_Osmancik.arff",
    "output_path": r"riceclf_output",
    "population_size": 100,
    "crossover_probability": 0.7,
    "mutation_probability": 0.2,
    "number_of_generations": 30,
    "elitism_size": 1,
}

# 'population_size': 700         # A smaller population for quicker testing
# 'crossover_probability': 0.9   # A slightly lower crossover to reduce disruption of individuals
# 'mutation_probability': 0.1    # A higher mutation rate to encourage diversity
# 'number_of_generations': 60    # Fewer generations to reduce computation time
# 'elitism_size': 2              # Minimal elitism to ensure the best solution is carried over


# Configure logging to display info level logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

parser = argparse.ArgumentParser("riceclf")

parser.add_argument(
    "-i",
    "--input",
    dest="dataset_path",
    type=str,
    default=CONFIG["dataset_path"],
    required=False,
    help=f"the path to the rice cammeo osmancik .arff dataset. [default: {CONFIG['dataset_path']}]",
)

parser.add_argument(
    "-d",
    "--destination",
    dest="output_path",
    type=str,
    default=CONFIG["output_path"],
    required=False,
    help=f"the directory path to save the outputs [default: {CONFIG['output_path']}]",
)

parser.add_argument(
    "-g",
    "--generations",
    dest="num_generations",
    type=int,
    default=CONFIG["number_of_generations"],
    required=False,
    help=f"the maximum number of generations for which the algorithm should run [default: {CONFIG['number_of_generations']}]",
)

parser.add_argument(
    "-p",
    "--population",
    dest="population_size",
    type=int,
    default=CONFIG["population_size"],
    required=False,
    help=f"the initial population size [default: {CONFIG['population_size']}]",
)

parser.add_argument(
    "-c",
    "--crossover-rate",
    dest="crossover_probability",
    type=float,
    default=CONFIG["crossover_probability"],
    required=False,
    help=f"the crossover rate [default: {CONFIG['crossover_probability']}]",
)

parser.add_argument(
    "-m",
    "--mutation-rate",
    dest="mutation_probability",
    type=float,
    default=CONFIG["mutation_probability"],
    required=False,
    help=f"the mutation rate [default: {CONFIG['mutation_probability']}]",
)

parser.add_argument(
    "-e",
    "--elites",
    dest="elitism_size",
    type=float,
    default=CONFIG["elitism_size"],
    required=False,
    help=f"the number of elite individuals to preserve across generations [default: {CONFIG['elitism_size']}]",
)

parser.add_argument(
    "--seed",
    dest="random_seed",
    type=int,
    default=None,
    required=False,
    help="the seed to fix the default random number generator [default: none]",
)


def main():
    """
    The main function to execute the GP algorithm for the rice classification task.
    """
    args = parser.parse_args()

    random.seed(args.random_seed)
    # Initialize the data handler and preprocess the data
    data_handler = DataHandler(args.dataset_path)
    data_handler.preprocess()

    # Check if the data was successfully preprocessed
    if data_handler.X_train is None or data_handler.y_train is None:
        logging.error("Data preprocessing failed. Exiting program.")
        return

    # Initialize and run the genetic programming engine
    gp_engine = GPEngine(
        data_handler,
        population_size=args.population_size,
        crossover_probability=args.crossover_probability,
        mutation_probability=args.mutation_probability,
        number_of_generations=args.num_generations,
        elitism_size=args.elitism_size,
    )
    population, logbook, hof, test_accuracy = gp_engine.run()

    # Plot the evolution of fitness over generations
    fitness_filename = os.path.join(args.output_path, "fitness_over_generations.png")
    gp_engine.plot_fitness(logbook, filename=fitness_filename)

    # Save the best individual to a text file
    best_individual_filename = os.path.join(args.output_path, "best_individual.txt")
    gp_engine.save_results(hof, filename=best_individual_filename)

    # Log the test accuracy of the best individual
    logging.info(f"Test Accuracy of the best individual: {test_accuracy:.2f}")


if __name__ == "__main__":
    main()
