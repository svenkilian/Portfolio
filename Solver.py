from __future__ import print_function

import time

import Functions as Fc
import Portfolio as Pf
from numpy.linalg import norm
from gurobipy import *
from pyDOE import *



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
    pf = Pf.Portfolio(solver='analytical', eta=eta, delta=delta, h=h, nwsum=nwsum, opt_type=opt_type, verbose=False)

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
                f1, f2 = Fc.obj_value(pf, solution)
            else:
                f1, f2 = Fc.obj_eff(pf, solution, delta, h)

        pf.portfolio_dec[iteration, :] = solution
        pf.portfolio_obj[iteration] = [f1, f2]  # Save individual objective values in Problem Instance
        end_run = time.time()
        diff = end_run - begin_run
        Fc.print_progress(iteration + 1, pf.nwsum, prog='Iter. avg: %g' % round(diff, 2), time_lapsed=end_run - begin_sim)
    end_sim = time.time()
    print('Simulation time: %g seconds' % (end_sim - begin_sim))

    z1 = pf.portfolio_obj[:, 0]  # List for plotting returns
    z2 = pf.portfolio_obj[:, 1]  # List for plotting risk

    p1 = z1[Fc.pareto_set(pf, pf.nwsum)]
    p2 = z2[Fc.pareto_set(pf, pf.nwsum)]

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
    pf = Pf.Portfolio(eta=eta, delta=delta, h=h, solver='random_search', opt_type=opt_type,
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
                    obj_effective = np.asarray(Fc.obj_eff(pf, np.asarray([w1, w2, w3]), pf.delta, pf.h))
                    obj_val = np.asarray(Fc.obj_value(pf, np.asarray([w1, w2, w3])))
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
        Fc.print_progress(run + 1, pf.nwsum, prog='Iter. avg: %g' % round(diff, 2), time_lapsed=end_run - begin_sim)

    end_sim = time.time()
    print('Simulation time: %g seconds' % (end_sim - begin_sim))

    z1 = pf.portfolio_obj[:, 0]  # List for plotting returns
    z2 = pf.portfolio_obj[:, 1]  # List for plotting risk

    p1 = z1[Fc.pareto_set(pf, pf.nwsum)]
    p2 = z2[Fc.pareto_set(pf, pf.nwsum)]

    return p1, p2, pf, begin_sim, end_sim


def solve_GA(opt_type='non_robust', delta=0.2, h=100, eta=0.03, selective_pressure=0.5, mutation_r=0.8,
             verbose=False):
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
    pf = Pf.Portfolio(eta=eta, delta=delta, h=h, solver='genetic', opt_type=opt_type,
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
                obj_effective = np.array(Fc.obj_eff(pf, offspring[:, i], pf.delta, pf.h))
                obj_val = np.array(Fc.obj_value(pf, offspring[:, i]))
                diff = obj_effective - obj_val
                # Calculate test value
                test_value = float(norm(diff)) / norm(obj_val)

                if test_value > eta or any(offspring[:, i] < 0):
                    if verbose:
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
                        if verbose:
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
                    obj_effective = np.array(Fc.obj_eff(pf, offspring[:, i], pf.delta, pf.h))
                    obj_val = np.array(Fc.obj_value(pf, offspring[:, i]))
                    diff = obj_effective - obj_val
                    # Calculate test value
                    test_value = float(norm(diff)) / norm(obj_val)
                    if test_value > eta:
                        if verbose:
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
        Fc.print_progress(run + 1, pf.nwsum, prog='Iter. avg: %g' % round(diff, 2), time_lapsed=end_run - begin_sim)

    end_sim = time.time()
    print('Simulation time: %g seconds' % (end_sim - begin_sim))
    print('\n')
    # print('Solutions: \n')
    # print(pf.portfolio_dec)

    z1 = pf.portfolio_obj[:, 0]  # List for plotting returns
    z2 = pf.portfolio_obj[:, 1]  # List for plotting risk

    # print(pareto_set(pf))
    p1 = z1[Fc.pareto_set(pf, pf.nwsum)]
    p2 = z2[Fc.pareto_set(pf, pf.nwsum)]

    return p1, p2, pf, begin_sim, end_sim
