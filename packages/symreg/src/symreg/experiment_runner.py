import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .genetic_programming import GeneticProgramming


class ExperimentRunner:
    """
    This class handles running the experiments and plotting the results.
    It creates multiple instances of the GeneticProgramming class with different
    configurations to test and compare their performances.
    """

    def __init__(
        self,
        filepath,
        num_generations=30,
        population_size=500,
        result_dir="symreg_results",
    ):
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
        seeded_runs = [self.run(param_sets, seed=seed) for seed in seeds]
        df = self._average_seeded_runs(seeded_runs)
        self.plot(df)

    def _average_seeded_runs(self, dfs):
        combined_df = pd.concat(dfs)
        df = (
            combined_df.groupby(
                ["crossover_rate", "mutation_rate", "elites", "generations"]
            )[["avg_fitness", "best_fitness"]]
            .mean()
            .reset_index()
        )
        return df

    def run(self, param_sets, seed=0) -> pd.DataFrame:
        """
        Runs the experiments with various configurations and collects the results into a DataFrame.
        Adjusted to meet the specific requirements of the assignment.

        :param param_sets: Iterable[tuple[float, float, int]], a list of parameter configurations with which to run an experiment
        :param seed: Iterable[int], the random seed with which to perform runs
        """
        results = []
        for crossover_rate, mutation_rate, elites in param_sets:
            gp_inst = GeneticProgramming(
                self.filepath,
                seed,
                self.population_size,
                self.num_generations,
                hof_size=1,
            )
            _, log, _ = gp_inst.run(crossover_rate, mutation_rate, nelites=elites)

            gens, avgs, bests = log.select("gen", "avg", "min")
            for gen, avg, best in zip(gens, avgs, bests):
                results.append(
                    (
                        gen,
                        avg,
                        best,
                        crossover_rate,
                        mutation_rate,
                        elites,
                    )
                )
        return pd.DataFrame(
            {
                "generations": [r[0] for r in results],
                "avg_fitness": [r[1] for r in results],
                "best_fitness": [r[2] for r in results],
                "crossover_rate": [r[3] for r in results],
                "mutation_rate": [r[4] for r in results],
                "elites": [r[5] for r in results],
            }
        )

    def plot(self, df):
        dfgroups = df.groupby(["crossover_rate", "mutation_rate", "elites"])
        for (cxpb, mutpb, elites), groupdf in dfgroups:
            self.plot_results(groupdf, f"cx_{cxpb}_mut_{mutpb}_elite_{elites}")

    def plot_results(self, results, title_suffix):
        print("Plotting results...")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot Average Fitness on a logarithmic scale if it contains non-zero values
        color = "tab:blue"
        ax.set_xlabel("Generation")
        ax.set_ylabel("Average Fitness", color=color)
        ax.plot(
            results["generations"],
            results["avg_fitness"],
            label="Average Fitness",
            color=color,
        )
        ax.tick_params(axis="y", labelcolor=color)
        if np.all(results["best_fitness"] > 0):
            # Ensure all values are positive for log scale
            ax.set_yscale("log")
        else:
            # Annotate on the plot that best fitness achieved zero
            ax.annotate(
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
        ax.plot(
            results["generations"],
            results["best_fitness"],
            label="Best Fitness",
            color=color,
            linestyle="--",
        )

        plt.title(f"Fitness over Generations ({title_suffix})")

        # Set grid for both primary and secondary axes
        ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)

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
    runner.run_experiments([(0.9, 0.1, 2)], seeds=range(5))
    print("Finished running experiments.")
