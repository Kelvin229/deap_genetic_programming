import os

import matplotlib.pyplot as plt
import numpy as np
from .genetic_programming import GeneticProgramming


class ExperimentRunner:
    """
    This class handles running the experiments and plotting the results.
    It creates multiple instances of the GeneticProgramming class with different
    configurations to test and compare their performances.
    """

    def __init__(self, filepath, num_generations, population_size, result_dir):
        """
        Initializes the experiment runner.

        :param filepath: str, the path to the data file
        :param num_generations: int, the maximum number of generations for which the algorithm should run
        :param population_size: int, the initial population size
        :param result_dir: str, the path of the directory where results should be saved
        """
        self.filepath = filepath
        self.num_generations = num_generations
        self.population_size = population_size
        self.result_dir = result_dir
        print(f"Results will be saved in: {result_dir}")

    def run_experiments(self, param_sets, seeds=range(10)):
        """
        Runs the experiments with various configurations and collects the results.
        Adjusted to meet the specific requirements of the assignment.

        :param param_sets: Iterable[tuple[float, float, int]], a list of parameter configurations with which to run an experiment
        :param seeds: Iterable[int], a list of random seeds with which to perform runs
        """
        results = {}

        for crossover_rate, mutation_rate, elitism in param_sets:
            param_key = (crossover_rate, mutation_rate, elitism)
            if param_key not in results:
                results[param_key] = {
                    'avg_fitness': np.zeros(self.num_generations),
                    'best_fitness': np.full(self.num_generations, np.inf),
                    'generations': np.arange(self.num_generations)
                }
            for seed in seeds:
                gp_instance = GeneticProgramming(
                    self.filepath,
                    seed,
                    self.population_size,
                    self.num_generations,
                    hof_size=elitism
                )
                _, log, _ = gp_instance.run(crossover_rate, mutation_rate, elitism > 0)

                # Process the logbook
                for record in log:
                    gen = record['gen']
                    results[param_key]['avg_fitness'][gen] += record['avg'] / len(seeds)
                    results[param_key]['best_fitness'][gen] = min(results[param_key]['best_fitness'][gen],
                                                                  record['min'])

            # Plot the aggregated results
            self.plot_results(results[param_key], f"cx_{crossover_rate}_mut_{mutation_rate}_elite_{elitism}")

    def plot_results(self, results, title_suffix):
        print("Plotting results...")
        fig, ax1 = plt.subplots(figsize=(10, 8))

        # Plot Average Fitness on a logarithmic scale if it contains non-zero values
        color = "tab:blue"
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Average Fitness", color=color)
        ax1.plot(
            results["generations"],
            results["avg_fitness"],
            label="Average Fitness",
            color=color,
        )
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_yscale("log")
        if not np.all(results["best_fitness"] > 0):
            # Annotate on the plot that best fitness achieved zero
            ax1.annotate(
                "Perfect fitness achieved!",
                xy=(1, 0),
                xycoords="axes fraction",
                fontsize=12,
                xytext=(-5, 5),
                textcoords="offset points",
                ha="right",
                va="bottom",
            )

        # Create a second plot for the Best Fitness
        color = "tab:red"
        ax1.plot(
            results["generations"],
            results["best_fitness"],
            label="Best Fitness",
            color=color,
            linestyle="--",
        )

        plt.title(f"Fitness over Generations ({title_suffix})")

        # Set grid for both primary and secondary axes
        ax1.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)

        plt.tight_layout()

        filename = f"experiment_{title_suffix}.png"
        filepath = os.path.join(self.result_dir, filename)
        plt.savefig(filepath)
        print(f"Saved plot to {filepath}")
        plt.close(fig)

        print("Finished plotting.")


if __name__ == "__main__":
    print("Running experiments...")
    runner = ExperimentRunner("training_data.csv")
    runner.run_experiments()
    print("Finished running experiments.")
