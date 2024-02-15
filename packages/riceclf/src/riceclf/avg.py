import argparse
import sys

import pandas as pd


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "csvfiles",
        nargs="+",
        type=argparse.FileType("r"),
        help="paths to csv files to average up",
    )

    parser.add_argument(
        "-o",
        "--outfile",
        dest="output_file",
        type=argparse.FileType("w"),
        default=sys.stdout,
        required=False,
        help="the CSV filepath to save the aggregated data [default: stdout]",
    )


def exec_command(args: argparse.Namespace):
    columns = [
        "TrainAverageFitness",
        "TrainMaxFitness",
        "TrainAvgHits",
        "TestAverageFitness",
        "TestMaxFitness",
    ]
    df = pd.concat([pd.read_csv(file) for file in args.csvfiles])
    avgdf = df.groupby("Generation")[columns].mean().reset_index()
    avgdf.to_csv(args.output_file, index=False)
