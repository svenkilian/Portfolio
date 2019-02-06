from __future__ import print_function
import time

from numpy.random import random, rand

from config import *
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
                self.pret[col], self.pvar[col] = obj_eff(self, self.pwm[:, col], delta, h)
        else:
            for col in range(self.popsize):
                self.pret[col], self.pvar[col] = obj_eff(self, pop[:, col], delta, h)

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
                [obj_eff(self, initial_population[:, i], self.delta, self.h) for i in range(self.popsize)])
            obj_val = np.array([obj_value(self, initial_population[:, i]) for i in range(self.popsize)])
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
                    diff_new = np.array(obj_eff(self, initial_population[:, i], self.delta, self.h)) - np.array(
                        obj_value(self, initial_population[:, i]))
                    test_values[i] = norm(diff_new) / norm(obj_value(self, initial_population[:, i]))
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
                    cd = crowding_distance(self, [p_1, p_2], obj_vals)
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


def obj_value(pf, x):
    """
    Calculates objective function values for given solution x
    :param pf: Problem instance
    :param x: Solution
    :return: Objective values f1, f2
    """
    f1 = np.matmul(np.transpose(pf.mu), x)
    f2 = 0.5 * np.matmul(np.matmul(np.transpose(x), pf.sigma), x)
    return f1, f2


def obj_eff(pf, x, delta, h):
    """
    Calculates values of mean effective objective function for given solution x for robust optimization
    :param pf: Problem instance
    :param x: solution
    :param delta: Neighborhood size
    :param h: Number of samples
    :return:
    """
    lhc_sample = (lhs(3, h) - 0.5) * 2.0 * delta
    f1 = (1.0 / h) * sum(np.matmul(np.transpose(pf.mu), y) for y in (np.add(np.transpose(x), lhc_sample)))
    f2 = (1.0 / h) * sum(
        0.5 * np.matmul(np.matmul(np.transpose(y), pf.sigma), y) for y in (np.add(np.transpose(x), lhc_sample)))
    return f1, f2


def fronts(obj_val):
    solutions = set(range(obj_val.shape[0]))  # Remaining solutions in population
    obj_val = np.reshape(np.concatenate([obj_val[:, 0], - obj_val[:, 1]]), (obj_val.shape[0], obj_val.shape[1]),
                         order='F')
    fronts = []
    ranks = np.array(range(obj_val.shape[0]))
    front_n = 1
    while len(solutions) > 0:
        is_efficient = np.array([(i in solutions) for i in range(obj_val.shape[0])])
        for i, c in enumerate(obj_val):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(obj_val[is_efficient] > c,
                                                    axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        front = np.array(range(obj_val.shape[0]))[is_efficient]
        ranks[front] = front_n
        fronts.append(front)  # Append front to fronts
        solutions = solutions - set(front)  # Remove current front from remaining solutions
        front_n += 1
    return np.array(fronts), ranks


def crowding_distance(pf, front, obj_val):
    """
    Calculates crowding distance for given front
    :type front: ndarray
    """
    obj_values = obj_val[front, :]
    distance = np.zeros(len(front))
    obj_min_max = np.zeros((pf.dim_obj, 2))
    for obj in range(pf.dim_obj):
        obj_min_max[obj] = [np.min(obj_values[:, obj]), np.max(obj_values[:, obj])]  # Save
        sorted_m = np.argsort(obj_values[:, obj])
        distance[sorted_m[0]] = 100000000
        distance[sorted_m[len(sorted_m) - 1]] = 100000000
        for i in range(1, len(sorted_m) - 1):
            distance[sorted_m[i]] += (obj_values[sorted_m[i + 1], obj] - obj_values[sorted_m[i - 1], obj]) / (
                    obj_min_max[obj, 1] - obj_min_max[obj, 0])
    return distance


def pareto_set(pf, n):
    """
    Return pareto set from given population
    :param pf:
    :return:
    """
    is_efficient = np.ones(pf.portfolio_obj.shape[0], dtype=bool)  # Initialize mask
    obj = np.reshape(np.concatenate([pf.portfolio_obj[:, 0], - pf.portfolio_obj[:, 1]]), (n, pf.dim_obj),
                     order='F')

    for i, c in enumerate(obj):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(obj[is_efficient] > c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self

    pareto_solutions = np.array(range(n))[is_efficient]
    return pareto_solutions


def solve_analytical(delta=0.2, h=100, eta=1.285, nwsum=None, opt_type='non_robust'):
    """
    Solves Portfolio optimization analytically both robust and non-robust
    :param eta:
    :param nwsum:
    :param delta: Neighborhood size
    :param h: Number of samples
    :param opt_type: Robustness
    :return: Optimal objective values f1, f2, problem instance pf
    """
    pf = Portfolio(solver='analytical', eta=eta, delta=delta, h=h, nwsum=nwsum, opt_type=opt_type, verbose=False)

    pf.print_information()

    begin_sim = time.time()
    for iteration in range(pf.nwsum):
        begin_run = time.time()
        solution = np.zeros(pf.dim_dec)

        # Create model
        m = Model()

        # Add variables to model
        x = m.addVars(range(pf.dim_dec), vtype=GRB.CONTINUOUS, lb=0, name="Allocation")

        # Populate objective
        obj = QuadExpr()
        for i in range(pf.dim_dec):
            temp = 0
            for j in range(pf.dim_dec):
                temp = temp + pf.sigma[i, j] * x[j]
            obj += pf.mu[i] * x[i] - pf.betas[iteration] * 0.5 * x[i] * temp

        # Generate random Latin hypercube sampling
        lhc_sample = (lhs(3, h) - 0.5) * 2.0 * delta

        # Populate robust objective
        obj_robust = QuadExpr()
        for k in range(h):
            for i in range(pf.dim_dec):
                temp = 0
                for j in range(pf.dim_dec):
                    temp = temp + pf.sigma[i, j] * (x[j] + lhc_sample[k, j])
                obj_robust += (1.0 / h) * (pf.mu[i] * (x[i] + lhc_sample[k, i]) - pf.betas[iteration] * 0.5 * (
                        x[i] + lhc_sample[k, i]) * temp)

        # Set objective depending on opt_type
        if opt_type == 'non_robust':
            m.setObjective(obj, GRB.MAXIMIZE)
        else:
            m.setObjective(obj_robust, GRB.MAXIMIZE)

        # Set constraint
        m.addConstr(quicksum(x[i] for i in range(pf.dim_dec)) == 1, 'Wholeness_Constraint')

        # Optimize model
        m.Params.OutputFlag = 0
        m.optimize()  # Solve the linear model using the Gurobi solver

        # Save Solution
        if m.status == GRB.Status.OPTIMAL:
            for i in range(pf.dim_dec):
                solution[i] = x[i].x

            # Save optimal value
            pf.results_ws = m.objVal

            # Calculate individual objective values
            if opt_type == 'non_robust':
                f1, f2 = obj_value(pf, solution)
            else:
                f1, f2 = obj_eff(pf, solution, delta, h)

        pf.portfolio_dec[iteration, :] = solution
        pf.portfolio_obj[iteration] = [f1, f2]  # Save individual objective values in Problem Instance
        end_run = time.time()
        diff = end_run - begin_run
        print_progress(iteration + 1, pf.nwsum, prog='Iter. avg: %g' % round(diff, 2), time_lapsed=end_run - begin_sim)
    end_sim = time.time()
    print('Simulation time: %g seconds' % (end_sim - begin_sim))

    z1 = pf.portfolio_obj[:, 0]  # List for plotting returns
    z2 = pf.portfolio_obj[:, 1]  # List for plotting risk

    p1 = z1[pareto_set(pf, pf.nwsum)]
    p2 = z2[pareto_set(pf, pf.nwsum)]

    return p1, p2, pf


def solve_random_search(opt_type='non_robust', delta=0.2, h=100, eta=1.285):
    """
    Solves optimization problem by means of a random search optimization
    :param opt_type: Type of optimization
    :param delta: Neighborhood size
    :param h: Number of samples
    :param eta: Threshold for robustness type 2
    :return: z1, z2, pf, begin_sim, end_sim
    """
    pf = Portfolio(eta=eta, delta=delta, h=h, solver='random_search', opt_type=opt_type,
                   verbose=False)  # Create simulation instance

    robustness = 'Non-Robustness' if opt_type == 'non_robust' else \
        'Robustness Type I' if opt_type == 'robust' else \
            'Robustness Type II' if opt_type == 'robust_2' else None

    print('\nExecution of Random Search Algorithm under %s'
          '\nBetas = %d'
          '\nRuns/Beta = %d'
          '\nPopulation Size = %d'
          '\nDelta = %g'
          '\nH = %d' % (robustness, pf.nwsum, pf.nruns, pf.popsize, pf.delta, pf.h))
    # Main loop for GA iterations with different betas
    begin_sim = time.time()  # Record beginning time of simulation
    for run, beta in enumerate(pf.betas):
        begin_run = time.time()
        wbest = np.zeros((pf.dim_dec, pf.nruns))
        pcritvec = np.zeros((pf.dim_dec, pf.nruns))
        # pret = np.zeros(pf.popsize)
        # pvar = np.zeros(pf.popsize)
        for iteration in range(pf.nruns):
            if opt_type == 'robust':
                pcrit = pf.pcrit_robust(beta, pf.delta, pf.h)
            else:
                pcrit = pf.pcrit_non_robust(beta)

            # Selection of the best portfolio
            top, topi = np.max(pcrit), np.argmax(pcrit)  # Best objective value and corresponding index
            wnew = pf.pwm[:, topi]  # Best portfolio

            # Store the best portfolio and the optimal criterion value for each solve_random_search
            wbest[:, iteration] = wnew  # Store best portfolio
            pcritvec[:, iteration] = top  # Store best objective value

            # Random generation (mutation) of popsize-1 new portfolios
            pwnew = np.zeros((pf.dim_dec, pf.popsize))
            i = 0
            while i < (pf.popsize - 1):
                w1 = wnew[0] + np.random.random(1) * pf.const
                w2 = wnew[1] + np.random.random(1) * pf.const
                w3 = wnew[2] + np.random.random(1) * pf.const
                temp = w1 + w2 + w3

                # Normalize values
                w1 = w1 / temp
                w2 = w2 / temp
                w3 = w3 / temp

                pwnew[:, i] = [w1, w2, w3]  # Fill new population with mutations

                if opt_type == 'robust_2':
                    obj_effective = np.asarray(obj_eff(pf, np.asarray([w1, w2, w3]), pf.delta, pf.h))
                    obj_val = np.asarray(obj_value(pf, np.asarray([w1, w2, w3])))
                    test_value = norm(obj_effective - obj_val) / norm(obj_val)
                    if test_value > eta:
                        print('Constraint violation. Solution not feasible.')
                        i -= 1
                        # pwnew[:, i] = wnew

                i += 1

            # Best portfolio for the solve_random_search in the last column of new population matrix
            pwnew[:, -1] = wnew

            # Update population
            pf.pwm = pwnew

        # Store optimal portfolio/weights
        pf.portfolio_dec[run, :] = wnew

        # Store optimal objective values resulting from optimal portfolio
        pf.portfolio_obj[run, 0] = pf.pret[topi]
        pf.portfolio_obj[run, 1] = pf.pvar[topi]

        end_run = time.time()
        diff = end_run - begin_run
        print_progress(run + 1, pf.nwsum, prog='Iter. avg: %g' % round(diff, 2), time_lapsed=end_run - begin_sim)

    end_sim = time.time()
    print('Simulation time: %g seconds' % (end_sim - begin_sim))

    z1 = pf.portfolio_obj[:, 0]  # List for plotting returns
    z2 = pf.portfolio_obj[:, 1]  # List for plotting risk

    p1 = z1[pareto_set(pf, pf.nwsum)]
    p2 = z2[pareto_set(pf, pf.nwsum)]

    return p1, p2, pf, begin_sim, end_sim


def solve_GA(opt_type='non_robust', delta=0.2, h=100, eta=0.03, selective_pressure=0.5, mutation_r=0.8,
             verbose='silent'):
    """
    Solves optimization problem by means of a genetic algorithm
    :param selective_pressure: Proportion of population chosen into mating pool
    :param mutation_r: Mutation propensity
    :param opt_type: Type of optimization
    :param delta: Neighborhood size
    :param h: Number of samples
    :param eta: Threshold for robustness type 2
    :return: z1, z2, pf, begin_sim, end_sim
    """
    pf = Portfolio(eta=eta, delta=delta, h=h, solver='genetic', opt_type=opt_type,
                   verbose=verbose)  # Create simulation instance

    pf.print_information()

    # JOB: Main loop for GA iterations with different betas
    begin_sim = time.time()  # Record beginning time of simulation
    for run, beta in enumerate(pf.betas):
        begin_run = time.time()
        wbest = np.zeros((pf.dim_dec, pf.nruns))
        pcritvec = np.zeros(pf.nruns)
        for iteration in range(pf.nruns):
            if opt_type == 'robust':
                pcrit = pf.pcrit_robust(beta, pf.delta, pf.h)
            else:
                pcrit = pf.pcrit_non_robust(beta)

            # JOB: Selection of mating pool
            # Select the selection_pressure * 100% best
            # mating_pool = pf.pwm[:, np.argsort(pcrit)[:-(int(pf.popsize * selective_pressure) + 1):-1]]
            mating_pool = pf.tournament_selection(n_parents=pf.popsize, pcrit=pcrit)
            # Match parents from mating pool
            mating_pool_parents = np.random.choice(range(len(mating_pool)),
                                                   size=(int(0.5 * len(mating_pool)), 2), replace=False)

            # Generate set of random numbers for crossover
            random_numbers = np.random.rand(len(mating_pool_parents), pf.dim_dec)
            # Set crossover parameter
            alpha = 0.5
            # Calculate gamma from random numbers and alpha
            gamma = np.zeros((len(mating_pool_parents), pf.dim_dec))
            for i in range(gamma.shape[0]):
                for j in range(gamma.shape[1]):
                    gamma[i][j] = (1 + 2 * alpha) * random_numbers[i][j] - alpha

            # Initilize and create offspring through crossover operation
            offspring = np.zeros((pf.dim_dec, len(mating_pool)))

            # JOB: Perform crossover for type 2 robustness
            if opt_type == 'robust_2':
                i = 0
            flag = False
            while i < offspring.shape[1]:
                for j in range(pf.dim_dec):
                    if i < (offspring.shape[1] / 2):
                        if flag:
                            rn = np.random.rand()
                            offspring[j][i] = rn * pf.pwm[j, mating_pool_parents[i, 0]] + (
                                    1 - rn) * \
                                              pf.pwm[j, mating_pool_parents[i, 1]]
                        else:
                            offspring[j][i] = gamma[i][j] * pf.pwm[j, mating_pool_parents[i, 0]] + (
                                    1 - gamma[i][j]) * \
                                              pf.pwm[j, mating_pool_parents[i, 1]]
                    else:
                        if flag:
                            rn = np.random.rand()
                            offspring[j][i] = rn * pf.pwm[
                                j, mating_pool_parents[i - mating_pool_parents.shape[0], 0]] + (
                                                      1 - rn) * pf.pwm[
                                                  j, mating_pool_parents[i - mating_pool_parents.shape[0], 1]]
                        else:
                            offspring[j][i] = (1 - gamma[i - mating_pool_parents.shape[0]][j]) * pf.pwm[
                                j, mating_pool_parents[i - mating_pool_parents.shape[0], 0]] \
                                              + gamma[i - mating_pool_parents.shape[0]][j] * pf.pwm[
                                                  j, mating_pool_parents[i - mating_pool_parents.shape[0], 1]]
                # JOB: Check if child is feasible
                col_sum = np.sum(offspring[:, i])
                offspring[:, i] = offspring[:, i] / float(col_sum)  # Normalize solution
                obj_effective = np.array(obj_eff(pf, offspring[:, i], pf.delta, pf.h))
                obj_val = np.array(obj_value(pf, offspring[:, i]))
                diff = obj_effective - obj_val
                # Calculate test value
                test_value = float(norm(diff)) / norm(obj_val)

                if test_value > eta or any(offspring[:, i] < 0):
                    if verbose == 1:
                        print('Violation in Offspring Solution %d, Perform Re-Crossover  \n' % i)
                        if test_value > eta:
                            print('Threshold violation.')
                        else:
                            print('Non-negativity violation.')
                            print(offspring[:, i])
                    flag = True
                    continue
                else:
                    flag = False
                    i += 1

            # JOB: All but robust type 2 cases
            else:
                i = 0
                while i < offspring.shape[1]:
                    for j in range(pf.dim_dec):
                        if i < (offspring.shape[1] / 2):
                            offspring[j][i] = np.abs(
                                gamma[i][j] * pf.pwm[j, mating_pool_parents[i, 0]] + (1 - gamma[i][j]) * \
                                pf.pwm[j, mating_pool_parents[i, 1]])
                        else:
                            offspring[j][i] = np.abs((1 - gamma[i - mating_pool_parents.shape[0]][j]) * pf.pwm[
                                j, mating_pool_parents[i - mating_pool_parents.shape[0], 0]] \
                                                     + gamma[i - mating_pool_parents.shape[0]][j] * pf.pwm[
                                                         j, mating_pool_parents[i - mating_pool_parents.shape[0], 1]])
                    # Normalize offspring
                    col_sum = np.sum(offspring[:, i])
                    offspring[:, i] = offspring[:, i] / float(col_sum)

                    if any(offspring[:, i] < 0):
                        if verbose == 1:
                            print('Violation in Offspring Solution %d, Perform Re-Crossover \n' % i)
                    else:
                        i += 1

            # JOB: Mutate offspring
            i = 0
            while i < offspring.shape[1]:
                for j in range(pf.dim_dec):
                    if np.random.binomial(1, mutation_r):
                        while True:
                            temp = offspring[j][i]
                            offspring[j][i] += np.random.normal(0, 0.05)
                            if offspring[j][i] >= 0:  # Ensure feasibility with respect tro non-negativity constraint
                                break
                            else:
                                offspring[j][i] = temp

                if opt_type == 'robust_2':
                    # JOB: Check if child is feasible
                    col_sum = np.sum(offspring[:, i])
                    offspring[:, i] = offspring[:, i] / float(col_sum)  # Normalize solution
                    obj_effective = np.array(obj_eff(pf, offspring[:, i], pf.delta, pf.h))
                    obj_val = np.array(obj_value(pf, offspring[:, i]))
                    diff = obj_effective - obj_val
                    # Calculate test value
                    test_value = float(norm(diff)) / norm(obj_val)
                    if test_value > eta:
                        if verbose == 1:
                            print('Violation in Mutated Offspring Solution %d, Perform Re-Crossover \n' % i)
                        continue
                    else:
                        i += 1
                else:
                    # Normalize offspring
                    col_sum = np.sum(offspring[:, i])
                    offspring[:, i] = offspring[:, i] / float(col_sum)
                    i += 1

            # JOB: Combine parent population and offspring
            combined_pool = np.concatenate((pf.pwm, offspring), axis=1)
            temp_comb = np.copy(combined_pool)  # Save combined pool

            # JOB: Select next generation
            if opt_type == 'robust':
                pcrit = pf.pcrit_robust(beta=beta, delta=pf.delta, h=pf.h, pop=combined_pool)
            else:
                pcrit = pf.pcrit_non_robust(beta=beta, pop=combined_pool)
            # prev_gen = np.copy(pf.pwm)  # Save previous generation

            # Update population
            pf.pwm = combined_pool[:, np.argsort(pcrit)[:-int(pf.popsize + 1):-1]]

            # Determine best objective value and best solution
            top, topi = np.max(pcrit), np.argmax(pcrit)  # Best objective value and corresponding index
            best = temp_comb[:, topi]  # Best portfolio

            # Store the best portfolio and the optimal criterion value
            wbest[:, iteration] = best  # Store best portfolio
            pcritvec[iteration] = top  # Store best objective value

        # fig, ax = plt.subplots()
        # x = range(pf.nruns)
        #
        # ax.plot(pcritvec, label='GA: Non-Robust')
        # ax.plot([wbest[0, i] for i in range(wbest.shape[1])], label='x1')
        # ax.plot([wbest[1, i] for i in range(wbest.shape[1])], label='x2')
        # ax.plot([wbest[2, i] for i in range(wbest.shape[1])], label='x3')
        # legend = ax.legend(loc='best', shadow=True, fontsize='small', frameon=None, fancybox=True)
        # plt.show()

        # Store optimal portfolio/weights
        pf.portfolio_dec[run, :] = best

        # Store optimal objective values resulting from optimal portfolio
        pf.portfolio_obj[run, 0] = pf.pret[topi]
        pf.portfolio_obj[run, 1] = pf.pvar[topi]

        end_run = time.time()
        diff = end_run - begin_run
        print_progress(run + 1, pf.nwsum, prog='Iter. avg: %g' % round(diff, 2), time_lapsed=end_run - begin_sim)

    end_sim = time.time()
    print('Simulation time: %g seconds' % (end_sim - begin_sim))
    print('\n')
    # print('Solutions: \n')
    # print(pf.portfolio_dec)

    z1 = pf.portfolio_obj[:, 0]  # List for plotting returns
    z2 = pf.portfolio_obj[:, 1]  # List for plotting risk

    # print(pareto_set(pf))
    p1 = z1[pareto_set(pf, pf.nwsum)]
    p2 = z2[pareto_set(pf, pf.nwsum)]

    return p1, p2, pf, begin_sim, end_sim


def solve_nsga_2(opt_type='non_robust', n_runs=None, popsize=None, delta=0.2, h=100, eta=0.03, mutation_r=0.8,
                 verbose='silent', real_time=False):
    pf = Portfolio(n_runs=n_runs, popsize=popsize, eta=eta, delta=delta, h=h, nwsum=0, solver='nsga_2',
                   opt_type=opt_type,
                   verbose=verbose)  # Create simulation instance
    pf.print_information()

    # JOB: Main loop for NSGA-II iterations
    begin_sim = time.time()  # Record beginning time of simulation

    # Real-time plotting
    if real_time:
        axes = plt.gca()
        axes.set_xlim(10, 15)
        axes.set_ylim(0, 12)
        axes.margins(0.8, 1.0)
        # ---- Style plot
        plt.xlabel('Expected Return')
        plt.ylabel('Expected Risk')
        plt.title('Pareto Front with NSGA-II \n'
                  'Runs: %d, population size: %d, \n'
                  'Delta: %g, h: %d'
                  % (pf.nruns, pf.popsize, pf.delta, pf.h))

        fig = plt.gcf()
        fig.set_size_inches(10, 7)
        plt.margins(0.8, 1.0)
        scatter, = plt.plot([], [], '.', label='NSGA-II')
        legend = ax.legend(loc='best', shadow=True, fontsize='small', frameon=None,
                           fancybox=True)

    # JOB: Main Loop
    for iteration in range(pf.nruns):
        begin_run = time.time()
        if opt_type == 'robust':
            obj_val = np.array([obj_eff(pf, pf.pwm[:, i], pf.delta, pf.h) for i in range(pf.popsize)])
        else:
            obj_val = np.array([obj_value(pf, pf.pwm[:, i]) for i in range(pf.popsize)])

        # JOB: Select mating pool from population
        if iteration == 0:
            mating_pool = pf.tournament_selection(n_parents=pf.popsize, obj_vals=obj_val, ranks=fronts(obj_val)[1])
            # print('\nFirst Iteration')
        else:
            mating_pool = pf.tournament_selection(pf.popsize, obj_vals=obj_val, ranks=fronts(obj_val)[1], crowded=True)
            # print('\nIteration %d' % (iteration + 1))
        # Match parents from mating pool
        mating_pool_parents = np.random.choice(range(len(mating_pool)), size=((int(0.5 * len(mating_pool))), 2),
                                               replace=False)

        # JOB: Perform Crossover to generate offspring
        offspring = crossover(pf=pf, obj_val=obj_val, mating_pool_parents=mating_pool_parents, mating_pool=mating_pool,
                              eta=pf.eta, verbose=verbose, opt_type=opt_type)

        # JOB: Mutate offspring
        offspring = mutate(pf=pf, offspring=offspring, mutation_r=mutation_r, eta=pf.eta, verbose=verbose,
                           opt_type=opt_type)

        # JOB: Combine parent population and offspring
        combined_pool = np.concatenate((pf.pwm, offspring), axis=1)

        # JOB: Select next generation
        if opt_type == 'robust':
            obj_val = np.concatenate((obj_val, np.array(
                [obj_eff(pf, offspring[:, i], pf.delta, pf.h) for i in range(offspring.shape[1])])), axis=0)
        else:
            obj_val = np.concatenate((obj_val, np.array([obj_value(pf, offspring[:, i]) for i in range(
                offspring.shape[1])])), axis=0)

        # JOB: Update population based on rank and crowding distance
        new_pop = combined_pool[:, select_by_rank_and_distance(pf.popsize, obj_val)]
        pf.pwm = new_pop

        # JOB: Time Run and Print Progress
        end_run = time.time()
        diff = end_run - begin_run
        print_progress(iteration + 1, pf.nruns, prog='Iter. avg: %g' % round(diff, 2), time_lapsed=end_run - begin_sim)

        # JOB: Calculate objective values and plot
        if real_time:
            if opt_type == 'robust':
                objs = np.array([obj_eff(pf, new_pop[:, i], pf.delta, pf.h) for i in range(new_pop.shape[1])])
            else:
                objs = np.array([obj_value(pf, new_pop[:, i]) for i in range(new_pop.shape[1])])

            scatter.set_xdata(objs[:, 0])
            scatter.set_ydata(objs[:, 1])
            # legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True, fontsize='small', frameon=None,
            #                    fancybox=True)
            plt.draw()
            plt.pause(1e-25)
            time.sleep(1e-25)

    end_sim = time.time()
    if real_time:
        scatter.set_xdata([])
        scatter.set_ydata([])
    print('Simulation time: %g seconds' % (end_sim - begin_sim))
    print('\n')
    # Calculate objective values
    if opt_type == 'robust':
        objs = np.array([obj_eff(pf, new_pop[:, i], pf.delta, pf.h) for i in range(new_pop.shape[1])])
    else:
        objs = np.array([obj_value(pf, new_pop[:, i]) for i in range(new_pop.shape[1])])

    pf.portfolio_obj = objs  # Set problem objectives to calculated objectives
    pareto_solutions = pf.portfolio_obj[pareto_set(pf, pf.popsize)]  # Filter out pareto-sominant set
    z1 = pareto_solutions[:, 0]  # List for plotting returns
    z2 = pareto_solutions[:, 1]  # List for plotting risk

    # print(pareto_set(pf))
    # p1 = z1[pareto_set(pf)]
    # p2 = z2[pareto_set(pf)]

    # return p1, p2, pf, begin_sim, end_sim
    return z1, z2, pf, begin_sim, end_sim


def crossover(pf, obj_val, mating_pool_parents, mating_pool, eta, verbose, opt_type):
    """
    Perform crossover from mating pool to generate offspring
    :param mating_pool_parents:
    :param mating_pool:
    :param eta:
    :param verbose:
    :return: offspring
    """

    # Generate set of random numbers for crossover
    random_numbers = np.random.rand(len(mating_pool_parents), pf.dim_dec)
    # Set crossover parameter
    alpha = 0.5
    # Calculate gamma from random numbers and alpha
    gamma = np.zeros((len(mating_pool_parents), pf.dim_dec))
    for i in range(gamma.shape[0]):
        for j in range(gamma.shape[1]):
            gamma[i][j] = (1 + 2 * alpha) * random_numbers[i][j] - alpha

    # Initialize and create offspring through crossover operation
    offspring = np.zeros((pf.dim_dec, len(mating_pool)))

    # JOB: Perform crossover for type 2 robustness
    if opt_type == 'robust_2':
        i = 0
    flag = False
    while i < offspring.shape[1]:
        for j in range(pf.dim_dec):
            if i < (offspring.shape[1] / 2):
                if flag:
                    rn = np.random.rand()
                    offspring[j][i] = rn * pf.pwm[j, mating_pool_parents[i, 0]] + (
                            1 - rn) * \
                                      pf.pwm[j, mating_pool_parents[i, 1]]
                else:
                    offspring[j][i] = gamma[i][j] * pf.pwm[j, mating_pool_parents[i, 0]] + (
                            1 - gamma[i][j]) * \
                                      pf.pwm[j, mating_pool_parents[i, 1]]
            else:
                if flag:
                    rn = np.random.rand()
                    offspring[j][i] = rn * pf.pwm[
                        j, mating_pool_parents[i - mating_pool_parents.shape[0], 0]] + (
                                              1 - rn) * pf.pwm[
                                          j, mating_pool_parents[i - mating_pool_parents.shape[0], 1]]
                else:
                    offspring[j][i] = (1 - gamma[i - mating_pool_parents.shape[0]][j]) * pf.pwm[
                        j, mating_pool_parents[i - mating_pool_parents.shape[0], 0]] \
                                      + gamma[i - mating_pool_parents.shape[0]][j] * pf.pwm[
                                          j, mating_pool_parents[i - mating_pool_parents.shape[0], 1]]
        # JOB: Check if child is feasible
        col_sum = np.sum(offspring[:, i])
        offspring[:, i] = offspring[:, i] / float(col_sum)  # Normalize solution
        obj_effective = np.array(obj_eff(pf, offspring[:, i], pf.delta, pf.h))
        obj_val = np.array(obj_value(pf, offspring[:, i]))
        diff = obj_effective - obj_val
        # Calculate test value
        test_value = float(norm(diff)) / norm(obj_val)

        if test_value > eta or any(offspring[:, i] < 0):
            if verbose == 1:
                print('Violation in Offspring Solution %d, Perform Re-Crossover  \n' % i)
                if test_value > eta:
                    print('Threshold violation.')
                else:
                    print('Non-negativity violation.')
                    print(offspring[:, i])
            flag = True
            continue
        else:
            flag = False
            i += 1

    # JOB: All but robust type 2 cases
    else:
        i = 0
        while i < offspring.shape[1]:
            for j in range(pf.dim_dec):
                if i < (offspring.shape[1] / 2):
                    offspring[j][i] = np.abs(
                        gamma[i][j] * pf.pwm[j, mating_pool_parents[i, 0]] + (1 - gamma[i][j]) * \
                        pf.pwm[j, mating_pool_parents[i, 1]])
                else:
                    offspring[j][i] = np.abs((1 - gamma[i - mating_pool_parents.shape[0]][j]) * pf.pwm[
                        j, mating_pool_parents[i - mating_pool_parents.shape[0], 0]] \
                                             + gamma[i - mating_pool_parents.shape[0]][j] * pf.pwm[
                                                 j, mating_pool_parents[i - mating_pool_parents.shape[0], 1]])
            # Normalize offspring
            col_sum = np.sum(offspring[:, i])
            offspring[:, i] = offspring[:, i] / float(col_sum)

            if any(offspring[:, i] < 0):
                if verbose == 1:
                    print('Violation in Offspring Solution %d, Perform Re-Crossover \n' % i)
            else:
                i += 1
    return offspring


def mutate(pf, offspring, mutation_r, eta, verbose, opt_type):
    # JOB: Mutate offspring
    i = 0
    while i < offspring.shape[1]:
        for j in range(pf.dim_dec):
            if np.random.binomial(1, mutation_r):
                while True:
                    temp = offspring[j][i]
                    offspring[j][i] += np.random.normal(0, 0.05)
                    if offspring[j][i] >= 0:  # Ensure feasibility with respect tro non-negativity constraint
                        break
                    else:
                        offspring[j][i] = temp

        if opt_type == 'robust_2':
            # JOB: Check if mutated child is feasible
            col_sum = np.sum(offspring[:, i])
            offspring[:, i] = offspring[:, i] / float(col_sum)  # Normalize solution
            obj_effective = np.array(obj_eff(pf, offspring[:, i], pf.delta, pf.h))
            obj_val = np.array(obj_value(pf, offspring[:, i]))
            diff = obj_effective - obj_val
            # Calculate test value
            test_value = float(norm(diff)) / norm(obj_val)
            if test_value > eta:
                if verbose == 1:
                    print('Violation in Mutated Offspring Solution %d, Perform Re-Crossover \n' % i)
                continue
            else:
                i += 1
        else:
            # Normalize offspring
            col_sum = np.sum(offspring[:, i])
            offspring[:, i] = offspring[:, i] / float(col_sum)
            i += 1
    return offspring


def select_by_rank_and_distance(n_select, obj_val):
    parents = set()
    pareto_fronts, ranks = fronts(obj_val)
    front_n = 0

    while len(parents) + pareto_fronts[front_n].size <= n_select:
        parents = parents.union(set(pareto_fronts[front_n]))
        front_n += 1
        if front_n == pareto_fronts.size:
            break
    if len(parents) < n_select:
        rem = n_select - len(parents)  # Remaining spots to fill
        front_by_cd = np.argsort(- crowding_distance(pf, pareto_fronts[front_n], obj_val))  # Sort front by cd (desc.)
        add_front = pareto_fronts[front_n][front_by_cd[:rem]]
        # print('Add front: ')
        # print(add_front)
        parents = parents.union(set(add_front))

    return list(parents)


def print_progress(iteration, total, prefix='', prog='', round_avg=0, suffix='', time_lapsed=0.0, decimals=1,
                   bar_length=100):
    """
    Call in a loop to create terminal progress bar
    :param iteration: current iteration (Int)
    :param total: total iterations (Int)
    :param prefix: prefix string (Str)
    :param suffix: suffix string (Str)
    :param decimals: positive number of decimals in percent complete (Int)
    :param bar_length: character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = u'\u258B' * filled_length + '-' * (bar_length - filled_length)
    pending_time = (time_lapsed / iteration) * (total - iteration)
    minutes = int(pending_time / 60)
    seconds = round(pending_time % 60)
    suffix = '%d mins, %g secs remaining' % (minutes, seconds)
    sys.stdout.write(
        '\r%s |%s| %s%s - Round %d of %d - %s - %s' % (prefix, bar, percents, '%', iteration, total, prog, suffix))
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    end_sim, begin_sim = 0, 0
    # JOB: Set Metaparameters
    h = 40
    delta = 0.28

    # JOB: Create Optimization Problem Instance
    pf = Portfolio(eta=0, delta=delta, h=h)
    data = dict()
    log_df = pd.DataFrame()

    # JOB: Run optimization and plot results
    fig, ax = plt.subplots()
    # legend = ax.legend(loc='best', shadow=True, fontsize='small', frameon=None,
    #                    fancybox=True)

    # ---- Run Non-Robust Optimization ----
    opt_type = 'non_robust'

    # z1, z2, pf, begin_sim, end_sim = solve_GA('non_robust', h=h)
    # ax.plot(z1, z2, '.', label='GA: Non-Robust')
    # data['ga_%s_%g' % (opt_type, 0)] = pf.portfolio_dec

    # z1, z2, pf, begin_sim, end_sim = solve_random_search('non_robust', h=h)
    # ax.plot(z1, z2, '.', label='Random Search: Non-Robust')
    # data['rs_%s_%g' % (opt_type, 0)] = pf.portfolio_dec

    z1, z2, pf = solve_analytical(opt_type='non_robust', nwsum=300)
    ax.plot(z1, z2, '.', label='Analytical: Non-Robust')
    data['an_%s_%g' % (opt_type, 0)] = pf.portfolio_dec

    z1, z2, pf, begin_sim, end_sim = solve_nsga_2(popsize=250, n_runs=2000, real_time=True)
    ax.plot(z1, z2, '.', label='NSGA-II: Non-Robust')
    data['nsga_II_%s_%g' % ('non_robust', 0)] = pf.pwm.transpose()

    print('Finished Non-Robust Optimization Runs')

    # ---- Run Robust Optimization of Type I ----
    opt_type = 'robust'
    delta = np.linspace(0.1, 0.25, 2)
    for d in delta:
        # z1, z2, pf, begin_sim, end_sim = solve_GA('robust', h=h, delta=d)
        # ax.plot(z1, z2, '.', label='GA: Robust, Delta=%g' % d)
        # data['ga_%s_%g' % (opt_type, d)] = pf.portfolio_dec
        # print([np.sum(pf.portfolio_dec[i, :]) for i in range(pf.nwsum)])
        #
        # z1, z2, pf, begin_sim, end_sim = solve_random_search('robust', h=h, delta=d)
        # ax.plot(z1, z2, '.', label='Random Search: Robust, Delta=%g' % d)
        # data['rs_%s_%g' % (opt_type, d)] = pf.portfolio_dec
        # print([np.sum(pf.portfolio_dec[i, :]) for i in range(pf.nwsum)])

        z1, z2, pf = solve_analytical(delta=d, opt_type='robust', h=h, nwsum=250)
        ax.plot(z1, z2, '.', label='Analytical: Robust, Delta=%g' % d)
        data['an_%s_%g' % (opt_type, d)] = pf.portfolio_dec
        # print([np.sum(pf.portfolio_dec[i, :]) for i in range(pf.nwsum)])

        z1, z2, pf, begin_sim, end_sim = solve_nsga_2(popsize=250, n_runs=2000, delta=d, h=h, opt_type='robust',
                                                      real_time=True)
        ax.plot(z1, z2, '.', label='NSGA-II: Robust, Delta=%g' % d)
        data['nsga_II_%s_%g' % (opt_type, 0)] = pf.pwm.transpose()

        # print('Finished Robust Optimization runs of Type I')
        pass

    # ---- Run Robust Optimization of Type II ----
    opt_type = 'robust_2'
    eta_s = np.linspace(0.022, 0.03, 1)

    for eta in eta_s:
        # z1, z2, pf, begin_sim, end_sim = solve_GA('robust_2', h=h, delta=delta, eta=eta, verbose=False)
        #     ax.plot(z1, z2, '.', label='GA: Robust Type II, Eta=%g' % eta)
        #     data['ga_%s_%g_%g' % (opt_type, d, eta)] = pf.portfolio_dec

        # z1, z2, pf = solve_analytical(opt_type='robust_2', h=h, delta=delta, eta=eta)
        # ax.plot(z1, z2, '.', label='Analytical: Robust Type II')
        # data['ga_%s_%g_%g' % (opt_type, d, eta)] = pf.portfolio_dec

        # z1, z2, pf, begin_sim, end_sim = solve_random_search('robust_2', h=h, delta=delta, eta=eta)
        # ax.plot(z1, z2, '.', label='Random Search: Robust Type II')
        # data['ga_%s_%g_%g' % (opt_type, d, eta)] = pf.portfolio_dec

        # z1, z2, pf, begin_sim, end_sim = solve_nsga_2(popsize=500, n_runs=1000, eta=eta, h=h, opt_type='robust_2',
        #                                               real_time=True, verbose=False)
        # ax.plot(z1, z2, '.', label='NSGA-II: Robust Type II, Delta=%g' % d)
        # data['nsga_II_%s_%g' % (opt_type, 0)] = pf.pwm.transpose()
        pass

    # print('Finished Type II Robustness')

    # ---- Save data in Excel File
    for i in data:
        df = pd.DataFrame(data[i], columns=[i + '_x' + str(j + 1) for j in range(pf.dim_dec)])
        log_df = pd.concat([log_df, df], join='outer', ignore_index=False, axis=1)

    # log_df.to_excel(ROOT_DIR + '\Data.xlsx')
    writer = pd.ExcelWriter(ROOT_DIR + '\Data.xlsx', engine='xlsxwriter')
    log_df.to_excel(writer, sheet_name='Data', header=True, index=True)
    worksheet = writer.sheets['Data']
    for idx, col in enumerate(log_df):
        series = log_df[col]
        max_len = max((series.astype(str).str.len().max(), len(str(series.name))))
        worksheet.set_column(idx + 1, idx + 1, max_len)
    writer.save()

    # ---- Style and save plot
    plt.xlabel('Expected Return')
    plt.ylabel('Expected Risk')
    plt.title('Pareto Front wrt. weighted sum for %g beta values, \n'
              'Runs per beta: %d, population size: %d, \n'
              'Delta: %g, h: %d, \n'
              'Last Simulation Time (sec): %g'
              % (pf.nwsum, pf.nruns, pf.popsize, pf.delta, pf.h, round(end_sim - begin_sim, 2)))
    # legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True, fontsize='small', frameon=None,
    #                    fancybox=True)
    legend = ax.legend(loc='best', shadow=True, fontsize='small', frameon=None,
                       fancybox=True)
    plt.show()
    # fig = plt.gcf()
    fig.savefig(ROOT_DIR + '\Diagram.png', bbox_inches='tight', dpi=400, quality=95)
    fig.savefig(ROOT_DIR + '\Diagram.pdf', bbox_inches='tight', dpi=400, quality=95)
    # show(block=False)
    # plt.close('all')
