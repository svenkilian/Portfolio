from __future__ import print_function

import Functions as Fc
from Functions import *
from Portfolio import *
from numpy.linalg import norm
from gurobipy import *
from pyDOE import *


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
        # ---- Style plot
        plt.title('Pareto Front with NSGA-II \n'
                  'Runs: %d, population size: %d, \n'
                  'Delta: %g, h: %d'
                  % (pf.nruns, pf.popsize, pf.delta, pf.h))

        scatter, = plt.plot([], [], '.', label='NSGA-II')
        legend = plt.gca().legend(loc='best', shadow=True, fontsize='small', frameon=None,
                                  fancybox=True)

    # JOB: Main Loop
    for iteration in range(pf.nruns):
        begin_run = time.time()
        if opt_type == 'robust':
            obj_val = np.array([Fc.obj_eff(pf, pf.pwm[:, i], pf.delta, pf.h) for i in range(pf.popsize)])
        else:
            obj_val = np.array([Fc.obj_value(pf, pf.pwm[:, i]) for i in range(pf.popsize)])

        # JOB: Select mating pool from population
        if iteration == 0:
            mating_pool = pf.tournament_selection(n_parents=pf.popsize, obj_vals=obj_val, ranks=Fc.fronts(obj_val)[1])
            # print('\nFirst Iteration')
        else:
            mating_pool = pf.tournament_selection(pf.popsize, obj_vals=obj_val, ranks=Fc.fronts(obj_val)[1],
                                                  crowded=True)
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
                [Fc.obj_eff(pf, offspring[:, i], pf.delta, pf.h) for i in range(offspring.shape[1])])), axis=0)
        else:
            obj_val = np.concatenate((obj_val, np.array([Fc.obj_value(pf, offspring[:, i]) for i in range(
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
                objs = np.array([Fc.obj_eff(pf, new_pop[:, i], pf.delta, pf.h) for i in range(new_pop.shape[1])])
            else:
                objs = np.array([Fc.obj_value(pf, new_pop[:, i]) for i in range(new_pop.shape[1])])

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
        scatter.remove()

    print('Simulation time: %g seconds' % (end_sim - begin_sim))
    print('\n')
    # Calculate objective values
    if opt_type == 'robust':
        objs = np.array([Fc.obj_eff(pf, new_pop[:, i], pf.delta, pf.h) for i in range(new_pop.shape[1])])
    else:
        objs = np.array([Fc.obj_value(pf, new_pop[:, i]) for i in range(new_pop.shape[1])])

    pf.portfolio_obj = objs  # Set problem objectives to calculated objectives
    pareto_solutions = pf.portfolio_obj[Fc.pareto_set(pf, pf.popsize)]  # Filter out pareto-sominant set
    z1 = pareto_solutions[:, 0]  # List for plotting returns
    z2 = pareto_solutions[:, 1]  # List for plotting risk

    # print(pareto_set(pf))
    # p1 = z1[pareto_set(pf)]
    # p2 = z2[pareto_set(pf)]

    # return p1, p2, pf, begin_sim, end_sim
    return z1, z2, pf, begin_sim, end_sim
