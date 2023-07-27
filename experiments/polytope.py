import random
import autograd.numpy as np
from src.feasible_region import lpnorm, ProbabilitySimplex, LpBall
from src.objective_function import SquaredLoss
from src.plotting import gap_plotter, determine_y_lims
from src.experiments_auxiliary_functions import run_experiment, create_reference_lines_automatically
from global_ import *
import matplotlib as mpl

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH


rhos = [0.05, 0.1, 0.25]
l = 4
for rho in rhos:
    y = np.zeros((DIMENSION, 1))
    y[0] = 1
    random_perturbation = np.abs(np.random.random((DIMENSION, 1)))
    random_perturbation[int(DIMENSION / 2):] = 0
    random_perturbation = random_perturbation / lpnorm(random_perturbation, 1)
    y = (1 - rho) * y + rho * random_perturbation
    offset = np.abs(np.random.random((DIMENSION, 1)))
    offset[int(DIMENSION / 2):] = 0
    offset = offset / lpnorm(offset, 1)
    y = y + offset
    y = 2 * y / lpnorm(y, 1)
    L = 1.
    mu = np.sqrt(2) / np.sqrt(DIMENSION)
    theta = 1 / 2
    r = theta
    # todo: check whether r = theta?
    m = rho /(4*mu)
    R = 0
    eta_R = l/(R+l)
    B = 6
    M = max(m*((2*B)/(eta_R))**(1-theta), (L+B)/(eta_R**2))

    M_0 = 4 * L

    M = max(M, M_0)

    A = np.identity(DIMENSION)

    objective_function = SquaredLoss(A=A, b=y)
    feasible_region = LpBall(dimension=DIMENSION, p=1)

    fw_step_size_rules = [{"step type": "open-loop", "a": l, "b": 1, "c": l, "d": 1}]
    primal_gaps, dual_gaps, best_gaps, _ = run_experiment(ITERATIONS, objective_function, feasible_region,
                                                          run_more=RUN_MORE,
                                                          fw_step_size_rules=fw_step_size_rules)

    gaps = [dual_gaps[0][1:ITERATIONS], best_gaps[0][1:ITERATIONS], primal_gaps[0][1:ITERATIONS]]
    labels = ["gap" + r'$_t$', "bestgap" + r'$_t$', "subopt" + r'$_t$']
    gap_0 = dual_gaps[0][0]
    gaps, labels, styles, colors, markers = create_reference_lines_automatically(gaps, labels, 1, l, gap_0)
    file_name = ("polytope_growth" +  "l1_ball_" + "_rho=" + str(rho) + "_l=" + str(l))

    k = min(1 / (1 - r), 2)
    S = 1
    eta_S = l / (S + l)
    val = eta_S*eta_R - eta_S**k*(1+2/M *(9*M/(2*m))**(1/(1-theta)))
    while val < 0:
        S += 1
        eta_S = l / (S + l)
        val = eta_S * eta_R - eta_S ** k * (1 + 2 / M * (9 * M / (2 * m)) ** (1 / (1 - theta)))

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
