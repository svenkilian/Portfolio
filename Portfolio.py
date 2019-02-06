from __future__ import print_function
import time
from numpy.random import random, rand

import Functions
import Functions as Fc
from config import *
from Solver import *
from Functions import *
import numpy as np
import pandas as pd

from numpy.linalg import norm
from gurobipy import *
import matplotlib.pyplot as plt
from pyDOE import *


class Portfolio:
    def __init__(self, n_runs=100, popsize=12, nwsum=200, eta=None, delta=None, h=None, solver=None,
                 opt_type='non_robust', verbose=False):
        """Constructor for Portfolio Optimization Problem instance containing problem problem parameters and
        storing results"""
        self.nruns = n_runs  # Number of runs of GA and random search
        self.popsize = popsize  # Population size
        self.dim_obj = 2  # Objective dimension
        self.dim_dec = 3  # Decision space dimension

        self.mu = np.asarray([8, 12, 15])  # Mean equity returns
        self.sigma = np.matrix([[6, -5, 4],  # Equity covariance matrix
                                [-5, 17, -11],
                                [4, -11, 24]])

        self.const = 0.1  # Mutation parameter
        self.eta = eta  # User-defined threshold for type II robustness
        self.delta = delta  # Neighborhood size
        self.h = h  # Number of sampling points
        self.solver = solver  # Solver used for simulation run
        self.opt_type = opt_type  # Type of robustness used in optimization

        self.pwm = self.initialize_population(self.opt_type, verbose)  # Generate initial population

        self.pret = np.zeros(self.popsize)  # Return values for current population
        self.pvar = np.zeros(self.popsize)  # Variance values for current population
        self.nwsum = nwsum  # Number of betas tested
        # self.betas = range(1, self.nwsum + 1)  # Array of tested betas
        self.betas = np.logspace(-1, 3, num=self.nwsum, base=10.0)  # Array of tested betas

        self.portfolio_dec = np.zeros((self.nwsum, self.dim_dec))  # Optimal portfolio allocation for all betas
        self.portfolio_obj = np.zeros((self.nwsum, self.dim_obj))  # Objective values for Return and Risk for all betas
        self.results_ws = np.zeros(self.nwsum)  # Result of weighted sum

    def pcrit_non_robust(self, beta, pop=None):
        """
        Implements fitness function (objective function) for Genetic Algorithm without robustness
        :param pop: Population (optional), Default: self.pwm
        :param beta: Weighting parameter
        :return: Fitness function values
        """
        if pop is None:
            self.pret = np.matmul(np.transpose(self.pwm), self.mu)  # Calculate returns for population
            self.pvar = np.diag(
                0.5 * np.matmul(np.matmul(np.transpose(self.pwm), self.sigma),
                                self.pwm))  # Calculate variance for population
        else:
            self.pret = np.matmul(np.transpose(pop), self.mu)  # Calculate returns for population
            self.pvar = np.diag(
                0.5 * np.matmul(np.matmul(np.transpose(pop), self.sigma),
                                pop))  # Calculate variance for population
        return self.pret - beta * self.pvar  # Calculate and return objective

    def pcrit_robust(self, beta, delta, h, pop=None):
        """
        Imlements fitness function (objective function) with type I robustness
        :param pop: Population (optional), Default: self.pwm
        :param beta: Weighting parameter
        :param delta: Size of neighborhood
        :param h: Number of samples
        :return:  Fitness function values
        """
        if pop is None:
            for col in range(self.popsize):
                self.pret[col], self.pvar[col] = Fc.obj_eff(self, self.pwm[:, col], delta, h)
        else:
            for col in range(self.popsize):
                self.pret[col], self.pvar[col] = Fc.obj_eff(self, pop[:, col], delta, h)

        return self.pret - beta * self.pvar

    def initialize_population(self, opt_type, verbose):
        """
        Initializes population randomly using Latin Hypercube Sampling
        :return:
        """
        random_set = lhs(3, self.popsize)  # Generate random sample through Latin Hypercube Sampling
        row_sum = np.asarray([sum(random_set[i, :]) for i in range(self.popsize)])
        initial_population = np.transpose(random_set / row_sum[:, None])  # Standardize initial solution population

        # JOB: In case of robustness of type II, ensure initial population's feasibility
        if opt_type == 'robust_2':
            obj_effective = np.array(
                [Functions.obj_eff(self, initial_population[:, i], self.delta, self.h) for i in range(self.popsize)])
            obj_val = np.array([Fc.obj_value(self, initial_population[:, i]) for i in range(self.popsize)])
            diff = obj_effective - obj_val

            test_values = np.array([float(norm(diff[i, :])) / float(norm(obj_val[i, :])) for i in range(self.popsize)])

            for i in range(self.popsize):
                while test_values[i] > self.eta:
                    if verbose:
                        print('Violation in Initial Solution %d \n' % i)
                        print('Test old: %g' % test_values[i])
                    initial_population[:, i] = lhs(3, 1)
                    col_sum = sum(initial_population[:, i])
                    initial_population[:, i] = initial_population[:,
                                               i] / col_sum  # Replacement with standardized solution
                    diff_new = np.array(Functions.obj_eff(self, initial_population[:, i], self.delta, self.h)) - np.array(
                        Fc.obj_value(self, initial_population[:, i]))
                    test_values[i] = norm(diff_new) / norm(Fc.obj_value(self, initial_population[:, i]))
                    if verbose == 1:
                        print('Test new: %g  \n' % test_values[i])
                        print(initial_population[:, i])
                        print('\n')

        return initial_population

    def tournament_selection(self, n_parents, pcrit=None, ranks=None, crowded=False, obj_vals=None):
        """
        Performs binary tournament selection and returns parent generation from population
        :return: parent generation
        """

        if pcrit is not None:  # Standard GA
            parent_gen = np.empty(n_parents)
            random_samples = np.random.choice(range(self.popsize), size=(n_parents, 2), replace=True)
            for i in range(n_parents):
                p_1 = random_samples[i, 0]
                p_2 = random_samples[i, 1]
                parent_gen[i] = p_1 if pcrit[p_1] > pcrit[p_2] else \
                    p_2 if pcrit[p_1] < pcrit[p_2] else \
                        random_samples[i, int(round(random()))]

        else:  # NSGA-II
            parent_gen = np.empty(n_parents)
            random_samples = np.random.choice(range(self.popsize), size=(n_parents, 2), replace=True)
            for i in range(n_parents):
                p_1 = random_samples[i, 0]
                p_2 = random_samples[i, 1]
                if crowded:  # In all but first round: crowded tournament selection
                    cd = Fc.crowding_distance([p_1, p_2], obj_vals)
                    parent_gen[i] = p_1 if ranks[p_1] < ranks[p_2] else \
                        p_2 if ranks[p_1] > ranks[p_2] else \
                            p_1 if cd[0] > cd[1] else \
                                p_2 if cd[0] < cd[1] else \
                                    random_samples[i, int(round(random()))]
                else:
                    parent_gen[i] = p_1 if ranks[p_1] > ranks[p_2] else \
                        p_2 if ranks[p_1] < ranks[p_2] else \
                            random_samples[i, int(round(random()))]

        return random_samples, parent_gen

    def print_information(self):
        robustness = 'Non-Robustness' if self.opt_type == 'non_robust' else \
            'Robustness Type I' if self.opt_type == 'robust' else \
                'Robustness Type II' if self.opt_type == 'robust_2' else None

        algorithm = 'Analytical Solver' if self.solver == 'analytical' else \
            'Random Search' if self.solver == 'random_search' else \
                'Genetic Algorithm' if self.solver == 'genetic' else \
                    'NSGA-II' if self.solver == 'nsga_2' else None

        print('\nExecution of %s under %s'
              '\nBetas = %d'
              '\nRuns/Beta = %d'
              '\nPopulation Size = %d'
              '\nDelta = %g'
              '\nH = %d'
              '\nEta = %g' % (
                  algorithm, robustness, self.nwsum, self.nruns, self.popsize, self.delta, self.h, self.eta))
