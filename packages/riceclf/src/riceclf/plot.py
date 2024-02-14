import argparse

import matplotlib.pyplot as plt
import pandas as pd


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "csvfile",
        type=argparse.FileType("r"),
        help="path of CSV file to visualize",
    )

    parser.add_argument(
        "pngfile",
        type=argparse.FileType("wb"),
        help="filepath of the plot PNG image to generate",
    )


def _plot_fitness(
    gens, train_avgfits, train_maxfits, test_avgfits, test_maxfits, filepath
):
    """
    Plots the fitness evolution over the generations.

    Args:
        filepath (str): The path to the file where the plot will be saved.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(gens, train_avgfits, label="Training Average Fitness")
    plt.plot(gens, train_maxfits, label="Training Maximum (Best) Fitness")
    plt.plot(gens, test_avgfits, label="Test Average Fitness")
    plt.plot(gens, test_maxfits, label="Test Maximum (Best) Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Fitness over Generations")
    plt.legend(loc="best")
    plt.savefig(filepath)
    plt.close()


def exec_command(args: argparse.Namespace):
    df = pd.read_csv(args.csvfile)
    _plot_fitness(
        df["Generation"],
        df["TrainAverageFitness"],
        df["TrainMaxFitness"],
        df["TestAverageFitness"],
        df["TestMaxFitness"],
        args.pngfile,
    )
