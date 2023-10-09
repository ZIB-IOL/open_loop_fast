import random
import autograd.numpy as np
from src.feasible_region import LpBall, lpnorm
from src.objective_function import SquaredLoss
from src.experiments_auxiliary_functions import run_experiment, create_reference_lines_automatically
from global_ import *
import matplotlib as mpl
from src.plotting import gap_plotter, determine_y_lims

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH

ps = [1.01, 1.5, 2.0, 2.5, 5.0]
ls = [2, 4]
lmbda = 0.2




for p in ps:
    fw_step_size_rules = []
    labels = []
    for l in ls:
        fw_step_size_rules.append({"step type": "open-loop", "a": l, "b": 1, "c": l, "d": 1})
        latex_string = r'$\eta_t = \frac{{{}}}{{{}}}$'.format(l, "t + " + str(l))
        labels.append(latex_string)
    fw_step_size_rules = fw_step_size_rules + [
        {"step type": "line-search"},
        {"step type": "log"}
    ]
    labels = labels + [
        "line-search",
        "log"
    ]
    q = 1 / (1 - 1 / p)
    assert p >= 1, "Only consider lp-balls with p>= 1."
    if p < 2:
        alpha = (p - 1) / 4
        r = 1.
        y = np.random.random((DIMENSION, 1))
        y = (lmbda + 1) * y / lpnorm(y, q)
        L = 1.
        M = L * (alpha * lmbda) ** (-1)
    elif p >= 2:
        alpha = 1 / (p*2**(p-1))
        r = 2 / p
        y = np.random.random((DIMENSION, 1))
        y = (lmbda + DIMENSION ** (1 / q - 1 / p)) * y / lpnorm(y, q)
        L = DIMENSION ** (1 / 2 - 1 / p)
        M = L * (alpha * lmbda) ** (-2 / p)

    M_0 = 4 * L
    M = max(M, M_0)

    A = np.identity(DIMENSION)

    objective_function = SquaredLoss(A=A, b=y)
    feasible_region = LpBall(dimension=DIMENSION, p=p)

    all_primal_gaps, all_dual_gaps, all_primal_dual_gaps, _ = run_experiment(ITERATIONS_MANY, objective_function,
                                                                             feasible_region,
                                                                             run_more=RUN_MORE_MANY,
                                                                             fw_step_size_rules=fw_step_size_rules)
    all_primal_gaps = [primal_gap[1:ITERATIONS_MANY] for primal_gap in all_primal_gaps]
    all_dual_gaps = [dual_gap[1:ITERATIONS_MANY] for dual_gap in all_dual_gaps]
    all_primal_dual_gaps = [primal_dual_gap[1:ITERATIONS_MANY] for primal_dual_gap in all_primal_dual_gaps]

    gap_0 = max([max(i) for i in all_dual_gaps])
    all_primal_gaps, labels, styles, colors, markers = create_reference_lines_automatically(
        all_primal_gaps, labels, None, None, gap_0, iterations=ITERATIONS_MANY, colors=COLORS_ALTERNATIVE_2)
    all_dual_gaps, _, _, _, _ = create_reference_lines_automatically(all_dual_gaps, labels, None, None, gap_0,
                                                                     iterations=ITERATIONS_MANY)
    all_primal_dual_gaps, _, _, _, _ = create_reference_lines_automatically(all_primal_dual_gaps, labels, None, None, gap_0,
                                                                            iterations=ITERATIONS_MANY)
    file_name = ("strong_growth_comparison" + "_l1_ball" + "_p=" + str(p) + "_lambda=" + str(lmbda))

    y_label = "subopt" + r"$_t$"
    gap_plotter(y_data=all_primal_gaps,
                labels=labels,
                iterations=ITERATIONS_MANY,
                file_name=("subopt_" + file_name),
                x_lim=(1, ITERATIONS_MANY),
                y_lim=determine_y_lims(all_primal_gaps),
                y_label=y_label,
                directory="experiments/figures/",
                legend=True,
                styles=styles,
                colors=colors,
                markers=markers
                )

    y_label = "gap" + r"$_t$"
    gap_plotter(y_data=all_dual_gaps,
                labels=labels,
                iterations=ITERATIONS_MANY,
                file_name=("gap_" + file_name),
                x_lim=(1, ITERATIONS_MANY),
                y_lim=determine_y_lims(all_dual_gaps),
                y_label=y_label,
                directory="experiments/figures/",
                legend=True,
                styles=styles,
                colors=colors,
                markers=markers
                )

    y_label = "primaldual" + r"$_t$"
    gap_plotter(y_data=all_primal_dual_gaps,
                labels=labels,
                iterations=ITERATIONS_MANY,
                file_name=("primaldual_" + file_name),
                x_lim=(1, ITERATIONS_MANY),
                y_lim=determine_y_lims(all_primal_dual_gaps),
                y_label=y_label,
                directory="experiments/figures/",
                legend=True,
                styles=styles,
                colors=colors,
                markers=markers
                )
