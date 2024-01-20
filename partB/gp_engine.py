import functools
import logging
import operator
import os
import random

import numpy as np
from deap import base, creator, tools, gp
from matplotlib import pyplot as plt


class GPEngine:
    def __init__(self, data_handler, population_size=700, crossover_probability=0.9, mutation_probability=0.1,
                 number_of_generations=60, elitism_size=2):
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
        return pset

    def _evalFitness(self, individual):
        func = self.toolbox.compile(expr=individual)
        predictions = np.array([np.clip(np.round(func(*row)), 0, 1) for row in self.data_handler.X_train.values])
        correct = np.sum(predictions == self.data_handler.y_train)  # Removed .values
        accuracy = correct / len(self.data_handler.y_train)
        return accuracy,

    def _setup_deap(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self._evalFitness)
        self.toolbox.register("select", tools.selTournament, tournsize=5)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

    def run(self):
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
            elites = sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)[:self.elitism_size]
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
        func = self.toolbox.compile(expr=best_ind)
        test_predictions = np.array([np.clip(np.round(func(*row)), 0, 1) for row in self.data_handler.X_test.values])
        correct_test = np.sum(test_predictions == self.data_handler.y_test)
        test_accuracy = correct_test / len(self.data_handler.y_test)

        logging.info(f"Test Accuracy of the best individual: {test_accuracy:.2f}")

        return pop, self.logbook, hof, test_accuracy

    def save_results(self, hof, filename="best_individual.txt"):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the file
        with open(filename, "w") as f:
            f.write(str(hof[0]))

    def plot_fitness(self, log, filename="fitness_over_generations.png"):
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        gen = log.select("gen")
        avg_fit = log.select("avg")
        min_fit = log.select("min")
        max_fit = log.select("max")

        plt.figure(figsize=(10, 5))
        plt.plot(gen, avg_fit, label="Average Fitness")
        plt.plot(gen, min_fit, label="Minimum Fitness")
        plt.plot(gen, max_fit, label="Maximum (Best) Fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Fitness over Generations")
        plt.legend(loc="best")
        plt.savefig(filename)
        plt.close()
