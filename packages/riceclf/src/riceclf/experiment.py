import pathlib
import sys
import argparse
import subprocess
import logging
import tempfile
import contextlib


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
        default=None,
        required=False,
        help="the path to the rice cammeo osmancik .arff dataset. [default: see `run` subcommand]",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        dest="output_dir",
        type=pathlib.Path,
        default="riceclf_experiment",
        required=False,
        help="the filepath to the directory where to save the aggregated data",
    )

    parser.add_argument(
        "-g",
        "--generations",
        dest="num_generations",
        type=int,
        default=None,
        required=False,
        help="the maximum number of generations for which the algorithm should run [default: see `run` subcommand]",
    )

    parser.add_argument(
        "-p",
        "--population",
        dest="population_size",
        type=int,
        default=None,
        required=False,
        help="the initial population size [default: see `run` subcommand]",
    )

    parser.add_argument(
        "-c",
        "--crossover-rate",
        dest="crossover_probability",
        type=float,
        default=None,
        required=False,
        help="the crossover rate [default: see `run` subcommand]",
    )

    parser.add_argument(
        "-m",
        "--mutation-rate",
        dest="mutation_probability",
        type=float,
        default=None,
        required=False,
        help="the mutation rate [default: see `run` subcommand]",
    )

    parser.add_argument(
        "-e",
        "--elites",
        dest="elitism_size",
        type=int,
        default=None,
        required=False,
        help="the number of elite individuals to preserve across generations [default: see `run` subcommand]",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=None,
        required=False,
    )

    parser.add_argument(
        "-s",
        "--seed",
        nargs="*",
        dest="random_seeds",
        type=int,
        default=[],
        required=False,
        help="the seeds with which to initialize random number generators for the experiment",
    )


def exec_avg(csvfiles, output):
    cmd = [sys.executable, "-m", "riceclf", "avg"] + [file.name for file in csvfiles]
    subprocess.run(cmd, stdout=output, stderr=sys.stderr)


def exec_command(args: argparse.Namespace):
    """
    The main function to execute the GP algorithm for the rice classification task.
    """
    dirpath: pathlib.Path = args.output_dir
    dirpath.mkdir(exist_ok=True, parents=True)
    run_cmd = [
        sys.executable,
        "-m",
        "riceclf",
        "run",
        "-o",
        dirpath,
        *(["-i", args.dataset_path] if args.dataset_path is not None else []),
        *(
            ["-g", str(args.num_generations)]
            if args.num_generations is not None
            else []
        ),
        *(
            ["-p", str(args.population_size)]
            if args.population_size is not None
            else []
        ),
        *(
            ["-c", str(args.crossover_probability)]
            if args.crossover_probability is not None
            else []
        ),
        *(
            ["-m", str(args.mutation_probability)]
            if args.mutation_probability is not None
            else []
        ),
        *(["-e", str(args.elitism_size)] if args.elitism_size is not None else []),
        "--verbose" if args.verbose else "--no-verbose",
    ]
    for seed in args.random_seeds:
        logging.info("Seed %s", seed)
        subprocess.run(
            run_cmd + ["--seed", str(seed)],
            stderr=sys.stderr,
        )
    with contextlib.ExitStack() as stack:
        csvfiles = [
            stack.enter_context(open(path, "r")) for path in dirpath.glob("run_*.csv")
        ]
        with open(dirpath.joinpath("result.csv"), "w") as output:
            exec_avg(csvfiles, output)
