import pandas as pd
import argparse
import io
import logging
import csv
import pathlib
import random
from deap import tools
from scipy.io import arff
from sklearn.model_selection import train_test_split


from .data_handler import DataHandler
from .gp_engine import GPEngine

# Configuration parameters
CONFIG = {
    "dataset_path": r"packages/riceclf/src/riceclf/Rice_Cammeo_Osmancik.arff",
    "population_size": 1000,
    "crossover_probability": 0.8,
    "mutation_probability": 0.2,
    "number_of_generations": 100,
    "elitism_size": 2,
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


def setup_parser(parser: argparse.ArgumentParser):
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
        "-o",
        "--outdir",
        dest="output_dir",
        type=pathlib.Path,
        default=pathlib.Path.cwd().joinpath("tmp"),
        required=False,
        help="the directory in which to save the results",
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
        type=int,
        default=CONFIG["elitism_size"],
        required=False,
        help=f"the number of elite individuals to preserve across generations [default: {CONFIG['elitism_size']}]",
    )

    parser.add_argument(
        "-s",
        "--seed",
        dest="random_seed",
        type=int,
        default=None,
        required=False,
        help="the seed to fix the default random number generator [default: none]",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        required=False,
    )


def _to_csv(logbook: tools.Logbook, csvfile: io.TextIOWrapper):
    gens = logbook.select("gen")
    train_avgfits, train_maxfits = logbook.chapters["fitness"].select("avg", "max")
    avgtrain_hits = logbook.chapters["hits"].select("avg")
    test_avgfits, test_maxfits = logbook.chapters["test_fitness"].select("avg", "max")
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "Generation",
            "TrainAverageFitness",
            "TrainMaxFitness",
            "TrainAvgHits",
            "TestAverageFitness",
            "TestMaxFitness",
        ]
    )
    for gen, avgfit_trn, maxfit_trn, avghits_trn, avgfit_tst, maxfit_tst in zip(
        gens, train_avgfits, train_maxfits, avgtrain_hits, test_avgfits, test_maxfits
    ):
        writer.writerow(
            [gen, avgfit_trn, maxfit_trn, avghits_trn, avgfit_tst, maxfit_tst]
        )


def exec_command(args: argparse.Namespace):
    """
    The main function to execute the GP algorithm for the rice classification task.
    """
    random.seed(args.random_seed)
    # Initialize the data handler and preprocess the data
    data_handler = DataHandler(args.dataset_path, random_state=args.random_seed)
    data_handler.preprocess()

    # Initialize and run the genetic programming engine
    gp_engine = GPEngine(
        data_handler,
        population_size=args.population_size,
        crossover_probability=args.crossover_probability,
        mutation_probability=args.mutation_probability,
        number_of_generations=args.num_generations,
        elitism_size=args.elitism_size,
        verbose=args.verbose,
    )
    _, logbook, hof, _ = gp_engine.run()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    with open(args.output_dir.joinpath(f"run_{args.random_seed}.csv"), "w") as csvfile:
        _to_csv(logbook, csvfile)

    with open(args.output_dir.joinpath(f"best_{args.random_seed}.fn"), "w") as file:
        file.write(str(hof[0]))

    data, _ = arff.loadarff(args.dataset_path)
    df = pd.DataFrame(data)
    X = df.drop("Class", axis=1)
    y = df["Class"].map({b"Cammeo": 0, b"Osmancik": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_seed
    )

    with open(
        args.output_dir.joinpath(f"test_{args.random_seed}.csv"), "w"
    ) as testfile:
        pd.concat(
            [
                pd.DataFrame(X_test, columns=X.columns),
                pd.Series(y_test, name=y.name),
            ],
            axis=1,
        ).to_csv(testfile, index=False)
