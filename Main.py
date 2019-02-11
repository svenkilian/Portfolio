from __future__ import print_function  ## Change ww
import time
from Functions import *
from config import *
from Solver import *
from NSGA_2 import *
from Functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyDOE import *

if __name__ == '__main__':
    end_sim, begin_sim = 0, 0
    # JOB: Set Metaparameters
    h = 100
    delta = 0.4

    # JOB: Create Optimization Problem Instance
    pf = Portfolio(eta=0, delta=delta, h=h)
    data = dict()
    log_df = pd.DataFrame()

    # JOB: Run optimization and plot results
    fig, ax = plt.subplots()
    # ---- Style plot ----
    legend = ax.legend(loc='best', shadow=True, fontsize='small', frameon=None,
                       fancybox=True)
    axes = plt.gca()
    axes.set_xlim(10, 15.5)
    axes.set_ylim(0, 14)
    axes.margins(0.8, 1.0)

    plt.xlabel('Expected Return')
    plt.ylabel('Expected Risk')
    plt.title('Pareto Front \n'
              'Runs: %d, population size: %d, \n'
              'Delta: %g, H: %d'
              % (pf.nruns, pf.popsize, pf.delta, pf.h))

    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.margins(0.8, 1.0)

    # ---- Run Non-Robust Optimization ----
    opt_type = 'non_robust'

    # z1, z2, pf, begin_sim, end_sim = solve_GA('non_robust', h=h)
    # ax.plot(z1, z2, '.', label='GA: Non-Robust')
    # data['ga_%s_%g' % (opt_type, 0)] = pf.portfolio_dec
    #
    # z1, z2, pf, begin_sim, end_sim = solve_random_search('non_robust', h=h)
    # ax.plot(z1, z2, '.', label='Random Search: Non-Robust')
    # data['rs_%s_%g' % (opt_type, 0)] = pf.portfolio_dec

    z1, z2, pf = solve_numerical(opt_type='non_robust', nwsum=400)
    ax.plot(z1, z2, '-', label='Numerically: Non-Robust')
    data['an_%s_%g' % (opt_type, 0)] = pf.portfolio_dec

    # z1, z2, pf, begin_sim, end_sim = solve_nsga_2(popsize=300, n_runs=100, crosstype='alpha', real_time=True)
    # ax.plot(z1, z2, '.', label='NSGA-II: Non-Robust')
    # data['nsga_II_%s_%g' % ('non_robust', 0)] = pf.pwm.transpose()

    # z1, z2, pf, begin_sim, end_sim = solve_nsga_2(popsize=200, n_runs=50, crosstype='simulated_binary', real_time=True,
    #                                               eta=np.inf)
    # ax.plot(z1, z2, '.', label='NSGA-II: Non-Robust')
    # data['nsga_II_%s_%g' % ('non_robust', 0)] = pf.pwm.transpose()

    print('Finished Non-Robust Optimization Runs')

    # ---- Run Robust Optimization of Type I ----
    opt_type = 'robust'
    delta = np.linspace(0.1, 0.4, 4)
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
        #
        # z1, z2, pf = solve_numerical(delta=d, opt_type='robust', h=h, nwsum=200)
        # ax.plot(z1, z2, '-', label='Numerically: Robust, Delta=%g' % d)
        # data['an_%s_%g' % (opt_type, d)] = pf.portfolio_dec
        # print([np.sum(pf.portfolio_dec[i, :]) for i in range(pf.nwsum)])

        # z1, z2, pf, begin_sim, end_sim = solve_nsga_2(popsize=400, n_runs=50, delta=d, h=h, opt_type='robust',
        #                                               real_time=True)
        # plt.figure(1)
        # ax.plot(z1, z2, '.', label='NSGA-II: Robust Type I, Delta=%g' % d)
        # data['nsga_II_%s_%g' % (opt_type, 0)] = pf.pwm.transpose()

        print('Finished Robust Optimization runs of Type I')
        pass

    # ---- Run Robust Optimization of Type II ----
    opt_type = 'robust_2'
    eta_s = np.linspace(0.3, 0.1, 4)

    for eta in eta_s:
        # z1, z2, pf, begin_sim, end_sim = solve_GA('robust_2', h=h, delta=delta, eta=eta, verbose=False)
        #     ax.plot(z1, z2, '.', label='GA: Robust Type II, Eta=%g' % eta)
        #     data['ga_%s_%g_%g' % (opt_type, d, eta)] = pf.portfolio_dec

        # z1, z2, pf, begin_sim, end_sim = solve_random_search('robust_2', h=h, delta=delta, eta=eta)
        # ax.plot(z1, z2, '.', label='Random Search: Robust Type II')
        # data['ga_%s_%g_%g' % (opt_type, d, eta)] = pf.portfolio_dec

        z1, z2, pf, begin_sim, end_sim = solve_nsga_2(popsize=200, n_runs=20, eta=eta, h=h, opt_type='robust_2',
                                                      crosstype='simulated_binary',
                                                      real_time=True, verbose=True, delta=0.35)

        plt.figure(1)
        ax.plot(z1, z2, '.', label='NSGA-II: Robust Type II, Delta=%g, Eta=%g' % (pf.delta, eta))
        data['nsga_II_%s_%g' % (opt_type, 0)] = pf.pwm.transpose()

        # feasible, constr_viol = is_feasible(pf, range(pf.popsize), verbose=False, obj_val=None)
        # feasibility_ratio = np.sum(feasible) / float(len(feasible))
        # avg_constr_viol = np.mean(constr_viol)
        # print('Feasibility Ratio: %g' % float(feasibility_ratio))
        # print('\n')
        # print('Average Constraint Violation: %g' % avg_constr_viol)
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
    plt.title('Pareto Front of Portfolio Optimization Problem, \n'
              'Number of Iterations: %d, \n'
              'Population Size: %d, \n'
              'Delta: %g, h: %d, \n'
              'Last Simulation Time (sec): %g'
              % (pf.nruns, pf.popsize, pf.delta, pf.h, round(end_sim - begin_sim, 2)))
    # legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True, fontsize='small', frameon=None,
    #                    fancybox=True)
    legend = ax.legend(loc='best', shadow=True, fontsize='medium', frameon=None,
                       fancybox=True)

    plt.show()
    # fig = plt.gcf()
    fig.savefig(ROOT_DIR + '\Diagram.png', bbox_inches='tight', dpi=400, quality=95)
    fig.savefig(ROOT_DIR + '\Diagram.pdf', bbox_inches='tight', dpi=400, quality=95)
    # show(block=False)
    # plt.close('all')
