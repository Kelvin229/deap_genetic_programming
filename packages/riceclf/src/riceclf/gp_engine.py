import logging
import random

import numpy as np
from deap import algorithms, base, creator, gp, tools
from scipy.special import expit as sigmoid


class GPEngine:
    """
    Encapsulates the setup and execution of a genetic programming process
    using DEAP for the rice classification task.

    Attributes: data_handler (DataHandler): An instance of the DataHandler class to manage data loading and preprocessing.
        population_size (int): The size of the genetic programming population.
        crossover_probability (float): Probability of crossover operation.
        mutation_probability (float): Probability of mutation operation.
        number_of_generations (int): The number of generations to run the genetic programming for.
        elitism_size (int): The number of top individuals to carry over to the next generation without changes.
        verbose (bool): Whether generation stats should be logged to outputs
    """

    def __init__(
        self,
        data_handler,
        population_size=100,
        crossover_probability=0.7,
        mutation_probability=0.2,
        number_of_generations=30,
        elitism_size=1,
        verbose=__debug__,
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
        fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
        test_fitness_stats = tools.Statistics(
            lambda ind: self._evalFitness(
                ind,
                X=self.data_handler.X_test,
                y=self.data_handler.y_test,
            )
        )
        depth_stats = tools.Statistics(lambda ind: ind.height)
        hit_stats = tools.Statistics(lambda ind: ind.hits)
        self.stats = tools.MultiStatistics(
            fitness=fitness_stats,
            test_fitness=test_fitness_stats,
            depth=depth_stats,
            hits=hit_stats,
        )
        self.stats.register("avg", np.mean)
        self.stats.register("max", np.max)

        self.verbose = verbose

    def _create_primitive_set(self):
        """
        Creates the primitive set for the genetic programming process.

        Returns:
            deap.gp.PrimitiveSet: The configured primitive set.
        """
        pset = gp.PrimitiveSetTyped(
            "MAIN", (float,) * len(self.data_handler.X_train.columns), float
        )
        pset.addPrimitive(np.add, (float, float), float, name="add")
        pset.addPrimitive(np.subtract, (float, float), float, name="sub")
        pset.addPrimitive(np.multiply, (float, float), float, name="mul")
        # pset.addPrimitive(
        #     lambda x, y: 0 if abs(y) <= 1e-04 else x / y,
        #     (float, float),
        #     float,
        #     name="protectedDiv",
        # )
        pset.addPrimitive(np.negative, (float,), float, name="neg")
        # pset.addPrimitive(np.sin, (float,), float)
        # pset.addPrimitive(np.cos, (float,), float)
        # pset.addPrimitive(np.less, (float, float), bool, name="lessThan")
        # pset.addPrimitive(np.allclose, (float, float), bool, name="eq")
        # pset.addPrimitive(np.logical_not, (bool,), bool, name="not_")
        # pset.addPrimitive(np.logical_and, (bool, bool), bool, name="and_")
        # pset.addPrimitive(np.logical_or, (bool, bool), bool, name="or_")
        # pset.addPrimitive(np.logical_xor, (bool, bool), bool, name="xor")
        # pset.addPrimitive(np.where, (bool, float, float), float, name="ifThenElse")
        # pset.addTerminal(True, bool, name="True")
        # pset.addTerminal(False, bool, name="False")

        def rand101():
            return random.uniform(-1, 1)

        pset.addEphemeralConstant("rand101", rand101, float)
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
        _, ncols = X.shape
        columns = [np.asarray(col).reshape(-1) for col in np.hsplit(X, ncols)]
        y_preds = np.round(sigmoid(func(*columns)))
        # Calculate accuracy
        accuracy = np.mean(y_preds == y)
        # register hits
        individual.hits = np.sum(y_preds == y)
        return (accuracy,)

    def _setup_deap(self):
        """
        Sets up the genetic programming environment using DEAP, including
        fitness, individuals, population, and the genetic operators.
        """
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create(
            "Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, hits=0
        )

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
        hof = tools.HallOfFame(1)

        pop, logbook = self.eaSimpleElitism(
            pop,
            self.toolbox,
            self.crossover_probability,
            self.mutation_probability,
            self.elitism_size,
            self.number_of_generations,
            self.stats,
            hof,
            self.verbose,
        )

        # Apply the compiled function to each row in the test set
        best_ind = hof[0]
        (test_accuracy,) = self.toolbox.evaluate(
            best_ind, X=self.data_handler.X_test, y=self.data_handler.y_test
        )

        logging.info("Test Accuracy of the best individual: %.2f", test_accuracy)
        logging.info("Tree depth of the best individual: %s", best_ind.height)
        logging.info("Best Individual: %s", best_ind)

        return pop, logbook, hof, test_accuracy

    def eaSimpleElitism(
        self,
        population,
        toolbox,
        cxpb,
        mutpb,
        nelites,
        ngen,
        stats=None,
        halloffame=None,
        verbose=__debug__,
    ):
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            logging.info(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population) - nelites)

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # include elites from previous generation
            elites = sorted(
                population, key=lambda ind: ind.fitness.values, reverse=True
            )[:nelites]
            offspring = offspring + elites

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                logging.info(logbook.stream)

        return population, logbook
