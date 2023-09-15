# References:
#   [1] Isabelle Guyon, Steve R. Gunn, Asa Ben-Hur, Gideon Dror, 2004. Result analysis of the NIPS 2003 feature
#   selection challenge. In: NIPS.


import os
import random
import autograd.numpy as np
from sklearn.preprocessing import StandardScaler

from src.feasible_region import LpBall
from src.objective_function import LogisticLoss
from src.plotting import gap_plotter, determine_y_lims
from src.experiments_auxiliary_functions import run_experiment, create_reference_lines_automatically
from global_ import *
import matplotlib as mpl

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH

ps = [1.]
l = 4
for p in ps:
    assert p >= 1, "Only consider lp-balls with p>= 1."
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
    primal_gaps, dual_gaps, primal_dual_gaps, _ = run_experiment(ITERATIONS_FEW, objective_function, feasible_region,
                                                          run_more=RUN_MORE_FEW,
                                                          fw_step_size_rules=fw_step_size_rules)

    gaps = [dual_gaps[0][1:ITERATIONS_FEW], primal_dual_gaps[0][1:ITERATIONS_FEW],
            primal_gaps[0][1:ITERATIONS_FEW]]
    labels = ["gap" + r'$_t$', "primaldual" + r'$_t$', "subopt" + r'$_t$']
    gap_0 = dual_gaps[0][0]
    gaps, labels, styles, colors, markers = create_reference_lines_automatically(gaps, labels, 1, l, gap_0)
    file_name = (("logistic_regression" + "_p=" + str(round(p, 2))) + "_l=" + str(l))

    gap_plotter(y_data=gaps,
                labels=labels,
                iterations=ITERATIONS_FEW,
                file_name=("gaps_" + file_name),
                x_lim=(1, ITERATIONS_FEW),
                y_lim=determine_y_lims(primal_gaps),
                y_label=("Optimality measure"),
                directory="experiments/figures/",
                legend=True,
                styles=styles,
                colors=colors,
                markers=markers
                )
