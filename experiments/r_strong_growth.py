# Uniformly convex feasible region, not strongly convex objective, and unconstrained optimum in the exterior experiment.


import random
import autograd.numpy as np

from src.feasible_region import LpBall, lpnorm
from src.objective_function import SquaredLoss
from src.plotting import primal_gap_plotter, determine_y_lims, only_min
from src.problem_settings import uniformly_convex
from src.experiments_auxiliary_functions import run_experiment
from global_ import *
import matplotlib as mpl

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH


ps = [2., 3., 5.]
lmbda = 2.
for p in ps:
    file_name = "r_strong_growth_lp_ball_p_=_" + str(p)

    A = np.identity(DIMENSION)
    y = np.random.random((DIMENSION, 1))
    y = y / lpnorm(y, p)

    objective_function = SquaredLoss(A=A, b=y)
    feasible_region = LpBall(dimension=DIMENSION, p=p)

    M = (p/lmbda)**(2/p)
    r = 2/p

    fw_step_size_rules = [
        # {"step type": "line-search"},
        # {"step type": "short-step"},
        {"step type": "open-loop", "a": 2, "b": 1, "c": 2, "d": 1},
        {"step type": "open-loop", "a": 4, "b": 1, "c": 4, "d": 1},
        {"step type": "open-loop", "a": 6, "b": 1, "c": 6, "d": 1},
    ]

    primal_gaps, labels = run_experiment(ITERATIONS, objective_function, feasible_region, run_more=RUN_MORE,
                                         fw_step_size_rules=fw_step_size_rules)
    primal_gaps = only_min(primal_gaps)

    # TODO: add theoretical rates....

    primal_gap_plotter(y_data=primal_gaps,
                       labels=labels,
                       iterations=ITERATIONS,
                       file_name=file_name,
                       x_lim=(1, ITERATIONS),
                       y_lim=determine_y_lims(primal_gaps),
                       y_label=r'$\mathrm{min}_i  \ h_i$',
                       directory="experiments/figures/non_polytope",
                       legend=legend
                       )

