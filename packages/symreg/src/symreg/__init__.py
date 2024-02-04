import argparse
import os
import sys

from .experiment_runner import ExperimentRunner

# Constants
NUM_GENERATIONS = 30
POPULATION_SIZE = 500
RESULT_DIR = r"symreg_results"
SEEDS = range(10)

# Parameter sets for experiments
PARAMETER_SETS = [
    (0.9, 0.1, 2),  # 90% crossover, 10% mutation, 2 elitism (configuration i)
    (1.0, 0.0, 2),  # 100% crossover, 0% mutation, 2 elitism (configuration ii)
    (0.0, 1.0, 2),  # 0% crossover, 100% mutation, 2 elitism (configuration iii)
    (0.9, 0.1, 0),  # Elitism test: same as (i) but with no elitism
]

parser = argparse.ArgumentParser(prog="Experiment Runner")

parser.add_argument(
    "-i",
    "--input",
    dest="input_filepath",
    type=str,
    default="training_data.csv",
    required=False,
    help="the path to the training data CSV file [default: training_data.csv]",
)

parser.add_argument(
    "-d",
    "--destination",
    dest="result_dir",
    type=str,
    default=RESULT_DIR,
    required=False,
    help=f"the path of the directory where results should be saved [default: {RESULT_DIR}]",
)

parser.add_argument(
    "-g",
    "--generations",
    dest="num_generations",
    type=int,
    default=NUM_GENERATIONS,
    required=False,
    help=f"the maximum number of generations for which the algorithm should run [default: {NUM_GENERATIONS}]",
)

parser.add_argument(
    "-p",
    "--population",
    dest="population_size",
    type=int,
    default=POPULATION_SIZE,
    required=False,
    help=f"the initial population size [default: {POPULATION_SIZE}]",
)


def main() -> int:
    args = parser.parse_args()

    print("Running experiments...")
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    runner = ExperimentRunner(
        args.input_filepath,
        num_generations=args.num_generations,
        population_size=args.population_size,
        result_dir=args.result_dir,
    )
    runner.run_experiments(PARAMETER_SETS, seeds=SEEDS)
    print("Finished running experiments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
