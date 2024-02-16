import sys
import argparse
import logging

from .run import setup_parser as setup_run_parser, exec_command as exec_run_command
from .avg import setup_parser as setup_avg_parser, exec_command as exec_avg_command
from .plot import setup_parser as setup_plot_parser, exec_command as exec_plot_command
from .experiment import (
    setup_parser as setup_experiment_parser,
    exec_command as exec_experiment_command,
)
from .test import setup_parser as setup_test_parser, exec_command as exec_test_command

# Configure logging to display info level logs
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

parser = argparse.ArgumentParser("riceclf")
subparser_creator = parser.add_subparsers(required=True)

run_parser = subparser_creator.add_parser("run")
setup_run_parser(run_parser)
run_parser.set_defaults(__exec=exec_run_command)

avg_parser = subparser_creator.add_parser("avg")
setup_avg_parser(avg_parser)
avg_parser.set_defaults(__exec=exec_avg_command)

plot_parser = subparser_creator.add_parser("plot")
setup_plot_parser(plot_parser)
plot_parser.set_defaults(__exec=exec_plot_command)

experiment_parser = subparser_creator.add_parser("experiment")
setup_experiment_parser(experiment_parser)
experiment_parser.set_defaults(__exec=exec_experiment_command)

test_parser = subparser_creator.add_parser("test")
setup_test_parser(test_parser)
test_parser.set_defaults(__exec=exec_test_command)


def main() -> int:
    """
    The main function to execute the GP algorithm for the rice classification task.
    """
    args = parser.parse_args()
    args.__exec(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
