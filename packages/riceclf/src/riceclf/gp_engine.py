import functools
import logging
import operator
import os
import random

import numpy as np
from deap import base, creator, tools, gp
from matplotlib import pyplot as plt


class GPEngine:
    """
    Encapsulates the setup and execution of a genetic programming process
    using DEAP for the rice classification task.

    Attributes:
        data_handler (DataHandler): An instance of the DataHandler class to manage data loading and preprocessing.
        population_size (int): The size of the genetic programming population.
        crossover_probability (float): Probability of crossover operation.
        mutation_probability (float): Probability of mutation operation.
        number_of_generations (int): The number of generations to run the genetic programming for.
        elitism_size (int): The number of top individuals to carry over to the next generation without changes.
    """

    def __init__(
        self,
        data_handler,
        population_size=100,
        crossover_probability=0.7,
        mutation_probability=0.2,
        number_of_generations=30,
        elitism_size=1,
    ):
        """
        Constructor for GPEngine.

        Args:
            data_handler (DataHandler): The data handler instance with preprocessed data.
            population_size (int): The size of the genetic programming population.
            crossover_probability (float): The probability of crossover between individuals.
            mutation_probability (float): The probability of mutation of individuals.
            number_of_generations (int): The number of generations to evolve.
            elitism_size (int): The number of best individuals to carry over to the next generation.
        """
        self.data_handler = data_handler
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.number_of_generations = number_of_generations
        self.elitism_size = elitism_size

        self.toolbox = base.Toolbox()
        self.pset = self._create_primitive_set()
        self._setup_deap()
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "nevals", "avg", "min", "max"
        self.verbose = True

    def _create_primitive_set(self):
        """
        Creates the primitive set for the genetic programming process.

        Returns:
            deap.gp.PrimitiveSet: The configured primitive set.
        """
        pset = gp.PrimitiveSet("MAIN", arity=self.data_handler.X_train.shape[1])
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(np.negative, 1)
        pset.addPrimitive(np.sin, 1)
        pset.addPrimitive(np.cos, 1)

        def rand101():
            return random.uniform(-1, 1)

        pset.addEphemeralConstant("rand101", functools.partial(rand101))
        pset.renameArguments(
            **{
                f"ARG{i}": column
                for i, column in enumerate(self.data_handler.X_train.columns)
            }
        )
        return pset

    def _evalFitness(self, individual, X=None, y=None):
        """
        Evaluates the fitness of an individual based on its accuracy of classifying
        the data.

        Args:
            individual (deap.creator.Individual): The individual to evaluate.
            X (pd.DataFrame): the data with which to evaluate the individual's fitness [defaults to the training data]
            y (pd.Series): the label of the data with which to evaluate the individual's fitness [defaults to the labels of the training data]

        Returns:
            tuple: A one-element tuple containing the accuracy of the individual on the data.
        """
        X = X if X is not None else self.data_handler.X_train
        y = y if y is not None else self.data_handler.y_train
        # Compile the individual's code to a function
        func = self.toolbox.compile(expr=individual)
        # Predict classes for the training set using the individual's function
        predictions = np.array(
            [np.clip(np.round(func(*row)), 0, 1) for row in X.values]
        )
        # Calculate the number of correct predictions
        correct = np.sum(predictions == y)
        # Calculate accuracy
        accuracy = correct / len(y)
        return (accuracy,)

    def _setup_deap(self):
        """
        Sets up the genetic programming environment using DEAP, including
        fitness, individuals, population, and the genetic operators.
        """
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self._evalFitness)
        self.toolbox.register("select", tools.selTournament, tournsize=5)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )

    def run(self):
        """
        Executes the genetic programming algorithm.

        Returns:
            tuple: A tuple containing the final population, the logbook with the recorded
            statistics, the hall of fame, and the accuracy of the best individual on the test set.
        """
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(self.elitism_size)

        # Evaluate the initial population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for gen in range(self.number_of_generations):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop) - self.elitism_size)
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_probability:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_probability:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The elite individuals are directly copied to the next generation
            elites = sorted(
                pop,
                key=lambda x: x.fitness.values[0],
                reverse=True,
            )[: self.elitism_size]
            pop[:] = offspring + elites

            # Update the hall of fame with the new population
            hof.update(pop)

            # Record the statistics
            record = self.stats.compile(pop) if self.stats is not None else {}
            self.logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if self.verbose:
                print(self.logbook.stream)

        # Apply the compiled function to each row in the test set
        best_ind = hof[0]
        (test_accuracy,) = self.toolbox.evaluate(
            best_ind, X=self.data_handler.X_test, y=self.data_handler.y_test
        )

        logging.info(f"Test Accuracy of the best individual: {test_accuracy:.2f}")

        return pop, self.logbook, hof, test_accuracy

    def save_results(self, hof, filename="best_individual.txt"):
        """
        Saves the best individual to a text file.

        Args:
            hof (deap.tools.HallOfFame): The hall of fame object containing the best individuals.
            filename (str): The path to the file where the best individual will be saved.
        """
        # Ensures the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the file
        with open(filename, "w") as f:
            f.write(str(hof[0]))

    def plot_fitness(self, log, filename="fitness_over_generations.png"):
        """
        Plots the fitness evolution over the generations.

        Args:
            log (deap.tools.Logbook): The logbook containing the statistics of the evolution.
            filename (str): The path to the file where the plot will be saved.
        """
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        gen = log.select("gen")
        avg_fit = log.select("avg")
        min_fit = log.select("min")
        max_fit = log.select("max")

        plt.figure(figsize=(10, 5))
        plt.plot(gen, avg_fit, label="Average Fitness")
        plt.plot(gen, min_fit, label="Minimum (Best) Fitness")
        plt.plot(gen, max_fit, label="Maximum Fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Fitness over Generations")
        plt.legend(loc="best")
        plt.savefig(filename)
        plt.close()
