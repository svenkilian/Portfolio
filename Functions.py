from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from Portfolio import *
from numpy.linalg import norm
from gurobipy import *

from Portfolio import *


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


def crossover(pf, obj_val, mating_pool_parents, mating_pool, eta, verbose, opt_type, crosstype='alpha'):
    """
    Perform crossover from mating pool to generate offspring
    Recombine parents using crossover for real-valued representations (Alpha-Blend Crossover)
    :param obj_val:
    :param crosstype:
    :param opt_type:
    :param mating_pool_parents:
    :param mating_pool:
    :param eta:
    :param verbose:
    :return: offspring
    """

    # Generate set of random numbers for crossover
    random_numbers = np.random.rand(len(mating_pool_parents), pf.dim_dec)
    # Initialize and create offspring through crossover operation
    offspring = np.zeros((pf.dim_dec, len(mating_pool)))

    # JOB: Crossover for crossover type 'Alpha Crossover'
    if crosstype == 'alpha':
        # Set crossover parameter
        alpha = 0.5
        # Calculate gamma from random numbers and alpha
        gamma = np.zeros((len(mating_pool_parents), pf.dim_dec))
        for i in range(gamma.shape[0]):
            for j in range(gamma.shape[1]):
                gamma[i][j] = (1 + 2 * alpha) * random_numbers[i][j] - alpha

        # JOB: Perform crossover for type 2 robustness
        if opt_type == 'robust_2':
            for i in range(offspring.shape[1]):
                for j in range(pf.dim_dec):
                    if i < (offspring.shape[1] / 2):
                        offspring[j][i] = np.abs(gamma[i][j] * pf.pwm[j, mating_pool_parents[i, 0]] + (
                                1 - gamma[i][j]) * \
                                                 pf.pwm[j, mating_pool_parents[i, 1]])
                    else:
                        offspring[j][i] = np.abs((1 - gamma[i - mating_pool_parents.shape[0]][j]) * pf.pwm[
                            j, mating_pool_parents[i - mating_pool_parents.shape[0], 0]] \
                                                 + gamma[i - mating_pool_parents.shape[0]][j] * pf.pwm[
                                                     j, mating_pool_parents[i - mating_pool_parents.shape[0], 1]])

        # JOB: All but robust type 2 cases
        else:
            for i in range(offspring.shape[1]):
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

    # JOB: Crossover for crossover type 'Simulated Binary Crossover'
    elif crosstype == 'simulated_binary':
        par_eta = 2
        co_betas = np.zeros((len(mating_pool_parents), pf.dim_dec))

        for i in range(len(mating_pool_parents)):
            for j in range(pf.dim_dec):
                co_betas[i][j] = (2 * random_numbers[i][j]) ** (1.0 / (par_eta + 1)) if np.random.rand() <= 0.5 else \
                    (1.0 / 2 * (1 - random_numbers[i][j])) ** (1.0 / (par_eta + 1))

        # JOB: Perform crossover for type 2 robustness
        if opt_type == 'robust_2':
            for i in range(offspring.shape[1]):
                for j in range(pf.dim_dec):
                    if i < (offspring.shape[1] / 2):
                        offspring[j][i] = np.abs(0.5 * ((1 + co_betas[i][j]) * (pf.pwm[j, mating_pool_parents[i, 0]]) +
                                                        (1 - co_betas[i][j]) * (pf.pwm[j, mating_pool_parents[i, 1]])))
                else:
                    offspring[j][i] = np.abs(0.5 * ((1 - co_betas[i - len(mating_pool_parents)][j]) * (
                        pf.pwm[j, mating_pool_parents[i - len(mating_pool_parents), 0]]) +
                                                    (1 + co_betas[i - len(mating_pool_parents)][j]) * (
                                                        pf.pwm[
                                                            j, mating_pool_parents[
                                                                i - len(mating_pool_parents), 1]])))

        # JOB: For all but robust type 2 cases
        else:
            for i in range(offspring.shape[1]):
                for j in range(pf.dim_dec):
                    if i < (offspring.shape[1] / 2):
                        offspring[j][i] = np.abs(0.5 * ((1 + co_betas[i][j]) * (pf.pwm[j, mating_pool_parents[i, 0]]) +
                                                        (1 - co_betas[i][j]) * (pf.pwm[j, mating_pool_parents[i, 1]])))
                    else:
                        offspring[j][i] = np.abs(0.5 * ((1 - co_betas[i - len(mating_pool_parents)][j]) * (
                            pf.pwm[j, mating_pool_parents[i - len(mating_pool_parents), 0]]) +
                                                        (1 + co_betas[i - len(mating_pool_parents)][j]) * (
                                                            pf.pwm[
                                                                j, mating_pool_parents[
                                                                    i - len(mating_pool_parents), 1]])))
                # Normalize offspring
                col_sum = np.sum(offspring[:, i])
                offspring[:, i] = offspring[:, i] / float(col_sum)

    return offspring


def mutate(pf, offspring, mutation_r, eta, verbose, opt_type):
    # JOB: Mutate offspring
    for i in range(offspring.shape[1]):
        for j in range(pf.dim_dec):
            if np.random.binomial(1, mutation_r):
                while True:
                    temp = offspring[j][i]
                    offspring[j][i] += np.random.normal(0, 0.15)
                    if offspring[j][i] >= 0:  # Ensure feasibility with respect tro non-negativity constraint
                        break
                    else:
                        offspring[j][i] = temp

        # Normalize offspring
        col_sum = np.sum(offspring[:, i])
        offspring[:, i] = offspring[:, i] / float(col_sum)

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
        front_by_cd = np.argsort(- crowding_distance(pareto_fronts[front_n], obj_val))  # Sort front by cd (desc.)
        add_front = pareto_fronts[front_n][front_by_cd[:rem]]
        # print('Add front: ')
        # print(add_front)
        parents = parents.union(set(add_front))

    return list(parents)


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


def crowding_distance(front, obj_val):
    """
    Calculates crowding distance for given front
    :type front: ndarray
    """
    obj_values = obj_val[front, :]
    distance = np.zeros(len(front))
    obj_min_max = np.zeros((obj_val.shape[1], 2))
    for obj in range(obj_val.shape[1]):
        obj_min_max[obj] = [np.min(obj_values[:, obj]), np.max(obj_values[:, obj])]
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


def is_feasible(pf, parents, verbose, obj_val):
    feasible = np.empty(len(parents), dtype=bool)
    constr_v = np.zeros(len(parents))
    for i, p in enumerate(parents):
        obj_effective = np.array(obj_eff(pf, pf.pwm[:, p], pf.delta, pf.h))
        if obj_val is None:
            obj_val = np.array(obj_value(pf, pf.pwm[:, p]))
        diff = obj_effective - obj_val
        # Calculate test value
        test_value = float(norm(diff)) / norm(obj_val)

        if test_value > pf.eta or any(pf.pwm[:, p] < 0):
            feasible[i] = False
            constr_v[i] = test_value - pf.eta
            if False:  # TODO: REVERSE
                print('Individual %d is not feasible.  \n' % p)
                if test_value > pf.eta:
                    print('Threshold violation.')
                else:
                    print('Non-negativity violation.')
                    print(pf.pwm[:, p])

        else:
            feasible[i] = True

    return feasible, constr_v
