import math
import operator
import random

import numpy
import pandas as pd
from deap import gp, creator, base, tools


class GeneticProgramming:
    """
    This class encapsulates the setup and execution of genetic programming.
    It uses the DEAP library to create and evolve a population based on given parameters.
    """

    def __init__(
        self, filepath, seed=0, population_size=300, num_generations=40, hof_size=1
    ):
        """
        Initializes the genetic programming environment.

        :param filepath: str, the path to the data file
        :param seed: int, the seed for random number generation
        :param population_size: int, the size of the population
        :param num_generations: int, the number of generations for evolution
        :param hof_size: int, the size of the hall of fame
        """
        self.filepath = filepath
        self.seed = seed
        self.population_size = population_size
        self.num_generations = num_generations
        self.hof_size = hof_size
        self.toolbox = base.Toolbox()
        self.setup_environment()
        self.load_training_points()

    def setup_environment(self):
        """
        Sets up the genetic programming environment, including the primitive set,
        types, operators, and evolutionary tools.
        """
        pset = gp.PrimitiveSet("MAIN", arity=1)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(operator.neg, 1)
        pset.addPrimitive(self.protectedDiv, 2)
        pset.addTerminal(1)
        pset.renameArguments(ARG0="x")

        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=pset)

        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register(
            "evaluate", lambda ind: self.evalSymbReg(ind, points=self.training_points)
        )

    def protectedDiv(self, left, right):
        """
        Performs division but with protection against division by zero.

        :param left: float, the numerator
        :param right: float, the denominator
        :return: float, the result of division or 1 if denominator is 0
        """
        if right == 0:
            return 1
        return left / right

    def load_training_points(self):
        """
        Loads training data from the specified file.
        """
        # Read the CSV file
        data = pd.read_csv(self.filepath)

        # Check if 'x' and 'y' columns are present
        if "x" not in data.columns or "y" not in data.columns:
            raise ValueError("CSV file must contain 'x' and 'y' columns")

        self.training_points = [(x, y) for x, y in zip(data["x"], data["y"])]

    def size_penalty(self, individual):
        """Penalize individuals for excessive size."""
        size_limit = 20
        if len(individual) > size_limit:
            return 1000.0  # A large penalty number
        else:
            return 0

    def evalSymbReg(self, individual, points):
        """
        Evaluates an individual's fitness using symbolic regression and includes a size penalty.

        :param individual: deap.gp.PrimitiveTree, the individual to evaluate
        :param points: list of tuples, the training data points
        :return: float, the mean squared error of the individual plus a size penalty if applicable
        """
        func = self.toolbox.compile(expr=individual)
        sqerrors = (round(func(x) - y, 3) ** 2 for x, y in points)
        mse = math.fsum(sqerrors) / len(points)
        size_penalty_value = self.size_penalty(individual)
        return (mse + size_penalty_value,)

    def run(self, crossover_rate, mutation_rate, nelites=0):
        """
        Runs the genetic programming algorithm.

        :param crossover_rate: float, the rate at which crossover is performed
        :param mutation_rate: float, the rate at which mutation is performed
        :return: tuple, the final population, the log, and the hall of fame
        """
        random.seed(self.seed)
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(self.hof_size) if self.hof_size > 0 else None
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("min", numpy.min)

        log = tools.Logbook()
        log.header = ["gen", "nevals"] + (stats.fields if stats else [])

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for gen in range(self.num_generations):
            offspring = self.toolbox.select(pop, len(pop) - nelites)

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # include elites from previous generation
            elites = sorted(pop, key=lambda ind: ind.fitness.values, reverse=True)[
                :nelites
            ]
            offspring = offspring + elites

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The next generation population will be the offspring
            pop[:] = sorted(offspring, key=lambda ind: ind.fitness.values)

            # If hof is not None, replace the worst with the best from the previous generation
            if hof is not None and len(hof) > 0:
                for i in range(min(self.hof_size, len(hof))):  # Access safely
                    pop[-(i + 1)] = hof[i]

            # Update the hall of fame with the generated individuals if hof is not None
            if hof is not None:
                hof.update(pop)

            # Update the statistics with the new population
            record = stats.compile(pop) if stats else {}
            log.record(gen=gen, nevals=len(invalid_ind), **record)
            print(log.stream)

        return pop, log, hof
