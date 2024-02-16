from deap import gp
import pandas as pd
import numpy as np
import argparse
import logging
import pathlib
from scipy.special import expit as sigmoid
from sklearn.metrics import confusion_matrix


from .data_handler import DataHandler
from .gp_engine import GPEngine


# Configure logging to display info level logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-i",
        "--input",
        dest="function_file",
        type=argparse.FileType("r"),
        required=True,
        help="the path to the serialized GP function",
    )

    parser.add_argument(
        "-f",
        "--testfile",
        dest="testfile",
        type=pathlib.Path,
        required=True,
        help="the path to the CSV file on which to execute the GP function",
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


def exec_command(args: argparse.Namespace):
    # Initialize and run the genetic programming engine
    df = pd.read_csv(args.testfile)
    pset = GPEngine._create_primitive_set(None, df.drop("Class", axis=1).columns)

    X = df.drop("Class", axis=1)
    y = df["Class"]
    with args.function_file as fnfile:
        individual = gp.PrimitiveTree.from_string(fnfile.read(), pset)
        (fitness,) = GPEngine._evalFitness(None, individual, pset=pset, X=X, y=y)
        print(fitness)
        func = gp.compile(individual, pset=pset)
        _, ncols = np.shape(X)
        columns = [np.asarray(col).reshape(-1) for col in np.hsplit(X, ncols)]
        y_preds = np.round(sigmoid(func(*columns)))

        print(confusion_matrix(y, y_preds))
