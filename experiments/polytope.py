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

p = 1.
l = 4
rhos = [0.05, 0.1, 0.25]
for rho in rhos:
    L_tilde = 1
    mu_tilde = np.sqrt(2)
    theta = 1/2
    diameter = 2
    L = L_tilde * diameter
    mu = mu_tilde**2 * rho**2
    B = 2 * diameter
    R = max(0, l/2 * float(np.sqrt(8*(L+B)/mu)))

    A = np.identity(DIMENSION)
    y = np.random.random((DIMENSION, 1))
    # y = np.ones(DIMENSION)
    y = np.zeros((DIMENSION, 1))
    y[0] = 1
    random_perturbation = np.abs(np.random.random((DIMENSION, 1)))
    random_perturbation[int(DIMENSION / 2):] = 0
    random_perturbation = random_perturbation / lpnorm(random_perturbation, 1)
    y = (1-rho) * y + rho * random_perturbation
    print(lpnorm(y, 1))
    assert np.abs(lpnorm(y, 1) - 1) <= 10**(-10)


    objective_function = SquaredLoss(A=A, b=y)
    feasible_region = LpBall(dimension=DIMENSION, p=p)

    fw_step_size_rules = [{"step type": "open-loop", "a": l, "b": 1, "c": l, "d": 1}]
    primal_gaps, dual_gaps, best_gaps, _ = run_experiment(ITERATIONS, objective_function, feasible_region,
                                                          run_more=RUN_MORE,
                                                          fw_step_size_rules=fw_step_size_rules)

    gaps = [dual_gaps[0][1:ITERATIONS], best_gaps[0][1:ITERATIONS], primal_gaps[0][1:ITERATIONS]]
    labels = ["gap" + r'$_t$', "bestgap" + r'$_t$', "subopt" + r'$_t$']
    gap_0 = dual_gaps[0][0]
    gaps, labels, styles, colors, markers = create_reference_lines_automatically(gaps, labels, 1, l, gap_0)
    file_name = (("polytope_growth" +  "_p=" + str(round(p, 2))) + "_rho=" + str(rho) + "_l=" + str(l))

    S = int(max(0, 2*R/l * np.sqrt(8*(L+B)/mu)))
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
