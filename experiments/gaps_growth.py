import random
import autograd.numpy as np
from src.feasible_region import LpBall, lpnorm
from src.objective_function import SquaredLoss
from src.plotting import gap_plotter, determine_y_lims
from src.experiments_auxiliary_functions import run_experiment, create_reference_lines_automatically
from global_ import *
import matplotlib as mpl

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH

rhos = [0.1, 0.5, 0.9]
ps = [1]

l = 4
for rho in rhos:
    for p in ps:
        assert p == 1, "Only the l1-ball is implemented at the moment."
        y = np.random.random((DIMENSION, 1))
        y = (1-rho) * y / lpnorm(y, p)
        L = 1.
        mu = np.sqrt(2) / np.sqrt(DIMENSION)
        theta = 1 / 2
        r = theta
        m = rho / mu
        M_0 = 4 * L

        A = np.identity(DIMENSION)

        objective_function = SquaredLoss(A=A, b=y)
        feasible_region = LpBall(dimension=DIMENSION, p=p)

        fw_step_size_rules = [{"step type": "open-loop", "a": l, "b": 1, "c": l, "d": 1}]
        primal_gaps, dual_gaps, best_gaps, _ = run_experiment(ITERATIONS, objective_function, feasible_region,
                                                              run_more=RUN_MORE,
                                                              fw_step_size_rules=fw_step_size_rules)

        gaps = [dual_gaps[0][1:ITERATIONS], best_gaps[0][1:ITERATIONS], primal_gaps[0][1:ITERATIONS]]
        labels = ["gap" + r'$_t$', "bestgap" + r'$_t$', "subopt" + r'$_t$']
        gap_0 = dual_gaps[0][0]
        gaps, labels, styles, colors, markers = create_reference_lines_automatically(gaps, labels, r, l, gap_0)
        file_name = (("gaps_growth" + "_r=" + str(round(r, 2)) + "_m=" + str(round(m, 2)) + "_p="
                      + str(round(p, 2))) + "_rho=" + str(rho) + "_l=" + str(l))


        # compute S
        # TODO: Get explicit formula for S at some point
        k = min(1 / (1 - r), 2)
        S = 1
        eta_S = l / (S+l)
        val = eta_S - eta_S**k*(1+2/M_0 *(9*M_0/(2*m))**(1/(1-r)))
        while val < 0:
            S += 1
            eta_S = l / (S + l)
            val = eta_S - eta_S**k*(1+2/M_0 *(9*M_0/(2*m))**(1/(1-r)))


        S_label = "S = " + str(S)
        lines = [(S, S_label)]

        gap_plotter(y_data=gaps,
                    labels=labels,
                    iterations=ITERATIONS,
                    file_name=("gaps_" + file_name),
                    x_lim=(1, ITERATIONS),
                    y_lim=determine_y_lims(primal_gaps),
                    y_label=("Optimality measure"),
                    directory="experiments/figures/",
                    legend=True,
                    styles=styles,
                    colors=colors,
                    markers=markers,
                    vertical_lines=lines
                    )
