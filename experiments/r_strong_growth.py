# Uniformly convex feasible region, not strongly convex objective, and unconstrained optimum in the exterior experiment.


import random
import autograd.numpy as np

from src.feasible_region import LpBall, lpnorm
from src.objective_function import SquaredLoss
from src.plotting import gap_plotter, determine_y_lims, only_min
from src.problem_settings import uniformly_convex
from src.experiments_auxiliary_functions import run_experiment, create_reference_line
from global_ import *
import matplotlib as mpl

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH


ps = [2., 3., 5.]
for p in ps:
    lmbda = 2.
    M = (p/lmbda)**(2/p)
    r = 2/p
    l = 6

    A = np.identity(DIMENSION)
    A[-1, -1] = 0.
    y = np.random.random((DIMENSION, 1))
    y = lmbda*y / lpnorm(y, p)

    objective_function = SquaredLoss(A=A, b=y)
    feasible_region = LpBall(dimension=DIMENSION, p=p)

    fw_step_size_rules = [{"step type": "open-loop", "a": l, "b": 1, "c": l, "d": 1}]
    primal_gaps, dual_gaps, best_gaps, _ = run_experiment(ITERATIONS, objective_function, feasible_region,
                                                               run_more=RUN_MORE, fw_step_size_rules=fw_step_size_rules)

    # TODO :  IMPLEMENT BESTGAP
    print("NEED TO IMPLEMENT BESTGAP")
    gaps = [primal_gaps[0][:ITERATIONS], dual_gaps[0][:ITERATIONS],
            [1/2*(dual_gaps[0][i]+primal_gaps[0][i]) for i in range(0, ITERATIONS)]] # TODO: HERE!
    labels = ["subopt" + r'$_t$', "gap" + r'$_t$', "bestgap" + r'$_t$']

    gap_0 = dual_gaps[0][0]

    if r < 1:
        if r < 1/2:
            gaps = [create_reference_line(ITERATIONS, gap_0, 1 / (1 - r)), create_reference_line(ITERATIONS, gap_0, 2),
                    create_reference_line(ITERATIONS, gap_0, 6)] + gaps
            labels = [('gap' + r'$_0 \cdot t^{-\frac{1}{1-r}}$'), ('gap' + r'$_0 \cdot t^{-2}$'),
                      ('gap' + r'$_0 \cdot t^{-\ell}$')] + labels
            styles = ["--", "-.", ":"] + STYLES
            colors = ["black", "black", "black"] + COLORS
            markers = ["", "", ""] + MARKERS
        if r >= 1/2:
            gaps = [create_reference_line(ITERATIONS, gap_0, 2), create_reference_line(ITERATIONS, gap_0, 1 / (1 - r)),
                    create_reference_line(ITERATIONS, gap_0, 6)] + gaps
            labels = [('gap' + r'$_0 \cdot t^{-2}$'), ('gap' + r'$_0 \cdot t^{-\frac{1}{1-r}}$'),
                      ('gap' + r'$_0 \cdot t^{-\ell}$')] + labels
            styles = ["-.", "--", ":"] + STYLES
            colors = ["black", "black", "black"] + COLORS
            markers = ["", "", ""] + MARKERS
    else:
        gaps = [create_reference_line(ITERATIONS, gap_0, 2), create_reference_line(ITERATIONS, gap_0, 6)] + gaps
        labels = [('gap' + r'$_0 \cdot t^{-2}$'), ('gap' + r'$_0 \cdot t^{-\ell}$')] + labels
        styles = ["-.", ":"] + STYLES
        colors = ["black", "black"] + COLORS
        markers = ["", ""] + MARKERS

    file_name = ("strong_growth_" + "r=" + str(round(r, 2)) + "_M=" + str(round(M, 2)) + "_p=" + str(round(p, 2)) +
                 "lmbda=" + str(round(lmbda, 2)))

    gap_plotter(y_data=gaps,
                labels=labels,
                iterations=ITERATIONS,
                file_name=("gaps_"+file_name),
                x_lim=(1, ITERATIONS),
                y_lim=determine_y_lims(primal_gaps),
                y_label=("Optimality measure"),
                directory="experiments/figures/strong_r_growth/",
                legend=True,
                styles=styles,
                colors=colors,
                markers=markers
                )




