# References:
#   [1] F. M. Harper and J. A. Konstan. The MovieLens datasets: History and context. ACM Transactions on Interactive
#   Intelligent Systems, 5(4):19:1–19:19, 2015.


import os
import random
import autograd.numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.feasible_region import LpBall, lpnorm, NuclearNormBall
from src.objective_function import SquaredLoss, LogisticLoss, HuberLossCollaborativeFiltering
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
radii = [1000, 3000]
for radius in radii:
    data = pd.read_csv(os.path.dirname(__file__) + '/../datasets/movielens100k.csv',
                       names=['user id', 'item id', 'rating', 'timestamp'])
    A = pd.pivot_table(data, values='rating', index='user id', columns='item id').values
    m, n = A.shape
    print("Dimensions: ", (m, n))

    objective_function = HuberLossCollaborativeFiltering(A=A)
    feasible_region = NuclearNormBall(m, n, radius=radius)

    fw_step_size_rules = [{"step type": "open-loop", "a": l, "b": 1, "c": l, "d": 1}]
    primal_gaps, dual_gaps, best_gaps, _ = run_experiment(ITERATIONS_HARD, objective_function, feasible_region,
                                                          run_more=RUN_MORE_HARD, fw_step_size_rules=fw_step_size_rules)

    gaps = [dual_gaps[0][1:ITERATIONS_HARD], best_gaps[0][1:ITERATIONS_HARD], primal_gaps[0][1:ITERATIONS_HARD]]
    labels = ["gap" + r'$_t$', "bestgap" + r'$_t$', "subopt" + r'$_t$']
    gap_0 = dual_gaps[0][0]
    gaps, labels, styles, colors, markers = create_reference_lines_automatically(gaps, labels, 1, l, gap_0)
    file_name = ("collaborative_filtering" +  "_radius=" + str(radius) + "_l=" + str(l))

    gap_plotter(y_data=gaps,
                labels=labels,
                iterations=ITERATIONS_HARD,
                file_name=("gaps_" + file_name),
                x_lim=(1, ITERATIONS_HARD),
                y_lim=determine_y_lims(primal_gaps),
                y_label=("Optimality measure"),
                directory="experiments/figures/",
                legend=True,
                styles=styles,
                colors=colors,
                markers=markers
                )
