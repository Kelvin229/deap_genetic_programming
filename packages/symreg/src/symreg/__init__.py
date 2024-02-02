import sys

from .experiment_runner import ExperimentRunner


def main() -> int:
    print("Running experiments...")
    runner = ExperimentRunner("training_data.csv")
    runner.run_experiments()
    print("Finished running experiments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
