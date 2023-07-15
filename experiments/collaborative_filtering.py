# References:
#   [1] F. M. Harper and J. A. Konstan. The MovieLens datasets: History and context. ACM Transactions on Interactive
#   Intelligent Systems, 5(4):19:1–19:19, 2015.


import os
import random
import autograd.numpy as np
from sklearn.preprocessing import StandardScaler

from src.feasible_region import LpBall, lpnorm
from src.objective_function import SquaredLoss, LogisticLoss
from src.plotting import gap_plotter, determine_y_lims
from src.experiments_auxiliary_functions import run_experiment, create_reference_line, \
    create_reference_lines_automatically
from global_ import *
import matplotlib as mpl

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH

l = 4
ps = [1.]
for p in ps:
    A = np.loadtxt(os.path.dirname(__file__) + '/../datasets/gisette/gisette_train.data')
    A = A[:2000, :]
    b = np.loadtxt(os.path.dirname(__file__) + '/../datasets/gisette/gisette_train.labels')
    b = b[:2000]

    scaler = StandardScaler()
    A = scaler.fit_transform(A)
    m, n = A.shape
    print("Dimensions: ", (m, n))

    objective_function = LogisticLoss(A=A, b=b)
    feasible_region = LpBall(dimension=A.shape[1], p=p)

    fw_step_size_rules = [{"step type": "open-loop", "a": l, "b": 1, "c": l, "d": 1}]
    primal_gaps, dual_gaps, best_gaps, _ = run_experiment(ITERATIONS, objective_function, feasible_region,
                                                          run_more=RUN_MORE,
                                                          fw_step_size_rules=fw_step_size_rules)

    gaps = [dual_gaps[0][1:ITERATIONS], best_gaps[0][1:ITERATIONS], primal_gaps[0][1:ITERATIONS]]
    labels = ["gap" + r'$_t$', "bestgap" + r'$_t$', "subopt" + r'$_t$']
    gap_0 = dual_gaps[0][0]
    gaps, labels, styles, colors, markers = create_reference_lines_automatically(gaps, labels, 1, l, gap_0)
    file_name = (("logistic_regression" +  "_p=" + str(round(p, 2))) + "_l=" + str(l))

    gap_plotter(y_data=gaps,
                labels=labels,
                iterations=ITERATIONS,
                file_name=("gaps_" + file_name),
                x_lim=(1, ITERATIONS),
                y_lim=determine_y_lims(primal_gaps),
                y_label=("Optimality measure"),
                directory="experiments/figures/logistic_regression/",
                legend=True,
                styles=styles,
                colors=colors,
                markers=markers
                )
