import os

import matplotlib.pyplot as plt
import numpy

from partA.genetic_programming import GeneticProgramming

NUM_GENERATIONS = 30
POPULATION_SIZE = 500


class ExperimentRunner:
    """
   This class handles running the experiments and plotting the results.
   It creates multiple instances of the GeneticProgramming class with different
   configurations to test and compare their performances.
   """

    def __init__(self, filename,
                 result_dir="C:\\Users\\Path\\to\\A1\\partA\\results"):
        """
       Initializes the experiment runner.

       :param filename: str, the path to the data file
       """
        self.filename = filename
        self.num_generations = NUM_GENERATIONS
        self.population_size = POPULATION_SIZE
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        print(f"Results will be saved in: {self.result_dir}")

    def run_experiments(self):
        """
        Runs the experiments with various configurations and collects the results.
        Adjusted to meet the specific requirements of the assignment.
        """
        seeds = range(10)
        num_runs = 10
        parameter_sets = [
            (0.9, 0.1, 2),  # 90% crossover, 10% mutation, 2 elitism (configuration i)
            (1.0, 0.0, 2),  # 100% crossover, 0% mutation, 2 elitism (configuration ii)
            (0.0, 1.0, 2),  # 0% crossover, 100% mutation, 2 elitism (configuration iii)
            (0.9, 0.1, 0),  # Elitism test: same as (i) but with no elitism
        ]
        results = {}

        for seed in seeds:
            for crossover_rate, mutation_rate, elitism in parameter_sets:
                param_key = (crossover_rate, mutation_rate, elitism)
                if param_key not in results:
                    results[param_key] = {
                        'avg_fitness': numpy.zeros(self.num_generations),
                        'best_fitness': numpy.full(self.num_generations, numpy.inf),
                        'generations': numpy.arange(self.num_generations)
                    }

                for run in range(num_runs):
                    seed = run
                    gp_instance = GeneticProgramming(
                        self.filename,
                        seed,
                        POPULATION_SIZE,
                        NUM_GENERATIONS,
                        hof_size=elitism
                    )
                    _, log, _ = gp_instance.run(crossover_rate, mutation_rate, elitism > 0)

                    # Process the logbook
                    for record in log:
                        gen = record['gen']
                        results[param_key]['avg_fitness'][gen] += record['avg'] / num_runs
                        results[param_key]['best_fitness'][gen] = min(results[param_key]['best_fitness'][gen],
                                                                      record['min'])

                # After all runs for a set of parameters, plot the aggregated results
                self.plot_results(results[param_key], f"cx_{crossover_rate}_mut_{mutation_rate}_elite_{elitism}")

    def plot_results(self, results, title_suffix):
        print("Plotting results...")
        fig, ax1 = plt.subplots(figsize=(10, 8))

        # Plot Average Fitness on a logarithmic scale if it contains non-zero values
        color = 'tab:blue'
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Average Fitness", color=color)
        ax1.plot(results['generations'], results['avg_fitness'], label='Average Fitness', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        if numpy.all(results['avg_fitness'] > 0):  # Ensure all values are positive for log scale
            ax1.set_yscale('log')

        # Create a second y-axis for the Best Fitness
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Best Fitness', color=color)
        ax2.plot(results['generations'], results['best_fitness'], label='Best Fitness', color=color, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        if numpy.all(results['best_fitness'] > 0):  # Ensure all values are positive for log scale
            ax2.set_yscale('log')
        else:
            # Annotate on the plot that best fitness achieved zero
            ax2.annotate('Perfect fitness achieved!', xy=(1, 0), xycoords='axes fraction', fontsize=12,
                         xytext=(-5, 5), textcoords='offset points',
                         ha='right', va='bottom')

        # Combine legends from both y-axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        plt.title(f"Fitness over Generations ({title_suffix})")

        # Set grid for both primary and secondary axes
        ax1.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        ax2.grid(False)  # Turn off grid for the secondary axis

        plt.tight_layout()

        filename = f"experiment_{title_suffix}.png"
        filepath = os.path.join(self.result_dir, filename)
        plt.savefig(filepath)
        print(f"Saved plot to {filepath}")
        plt.close(fig)
        plt.clf()

        print("Finished plotting.")


if __name__ == "__main__":
    print("Running experiments...")
    runner = ExperimentRunner("training_data.csv")
    runner.run_experiments()
    print("Finished running experiments.")
