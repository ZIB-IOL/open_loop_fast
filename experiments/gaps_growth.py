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

rhos = [0.3, 1.0, 1.7]
ps = [1]

l = 4
for rho in rhos:
    dist = (rho + 0.1)/2
    for p in ps:
        assert p == 1, "Only the l1-ball is implemented at the moment."
        good_vec = False
        while good_vec == False:
            y = np.zeros((DIMENSION, 1))
            y[0] = 1 - dist
            random_part = np.random.random((DIMENSION-1, 1))
            y[1:] = (dist-0.1)*random_part/lpnorm(random_part, 1)
            good_vec = (np.argmax(y) == 0)
        # y = np.random.random((DIMENSION, 1))
        # y = (1-rho) * y / lpnorm(y, p)
        assert lpnorm(y, 1) == 0.9
        assert np.argmax(y) == 0
        vertex = np.zeros((DIMENSION, 1))
        vertex[0] = 1
        print(lpnorm(y - vertex, 1))
        assert np.abs(lpnorm(y - vertex, 1) - rho) <= 10**(-6)
        L = 1.
        mu = np.sqrt(2) * np.sqrt(DIMENSION)
        theta = 1 / 2
        r = theta
        m = (rho) / mu
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
        k = min(1 / (1 - r), 2)
        S = int(max(1, np.ceil(l*((1 + 2/M_0 * (9*M_0/(2*m))**(1/(1-r)))/l)**(1/(k-1)) - l)))
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
