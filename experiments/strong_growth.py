import random
import autograd.numpy as np
from src.feasible_region import LpBall, lpnorm
from src.objective_function import SquaredLoss
from src.plotting import gap_plotter, determine_y_lims
from src.experiments_auxiliary_functions import run_experiment, create_reference_line
from global_ import *
import matplotlib as mpl

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH


ps = [1.1, 1.5, 2., 2.5, 3., 5., 7.]
for p in ps:
    lmbda = 2
    L = 1.
    l = 4
    if p >= 2:
        alpha = 1/p
        M = L*(p/lmbda)**(2/p)
        r = 2/p
    elif p < 2:
        alpha = (p-1)/2
        M = 1/(alpha*lmbda)
        r = 1.


    A = np.identity(DIMENSION)
    y = np.random.random((DIMENSION, 1))
    y = lmbda*y / lpnorm(y, p)

    objective_function = SquaredLoss(A=A, b=y)
    feasible_region = LpBall(dimension=DIMENSION, p=p)

    fw_step_size_rules = [{"step type": "open-loop", "a": l, "b": 1, "c": l, "d": 1}]
    primal_gaps, dual_gaps, best_gaps, _ = run_experiment(ITERATIONS, objective_function, feasible_region,
                                                               run_more=RUN_MORE, fw_step_size_rules=fw_step_size_rules)

    # TODO :  IMPLEMENT BESTGAP
    print("NEED TO IMPLEMENT BESTGAP")
    gaps = [primal_gaps[0][:ITERATIONS],
            [(1/2*dual_gaps[0][i]+1/2*primal_gaps[0][i]) for i in range(0, ITERATIONS-1)], dual_gaps[0][:ITERATIONS]]
    labels = ["subopt" + r'$_t$', "bestgap" + r'$_t$', "gap" + r'$_t$']

    gap_0 = dual_gaps[0][0]



    if r < 1:
        paras_unsorted = [1 /(1-r), 2, l]
        gaps_unsorted = [create_reference_line(ITERATIONS, gap_0, 1 / (1 - r)),
                         create_reference_line(ITERATIONS, gap_0, 2),
                         create_reference_line(ITERATIONS, gap_0, l)]
        labels_unsorted = [('gap' + r'$_0 \cdot t^{-\frac{1}{1-r}}$'),
                           ('gap' + r'$_0 \cdot t^{-2}$'),
                           ('gap' + r'$_0 \cdot t^{-\ell}$')]
        styles_unsorted = ["--", "-.", ":"]
        sorted_lists = sorted(zip(paras_unsorted, gaps_unsorted, labels_unsorted, styles_unsorted))
        paras_sorted, gaps_sorted, labels_sorted, styles_sorted = zip(*sorted_lists)
        # Convert the sorted tuples back to lists
        paras_sorted = list(paras_sorted)
        gaps_sorted = list(gaps_sorted)
        labels_sorted = list(labels_sorted)
        styles_sorted = list(styles_sorted)

        gaps = gaps_sorted + gaps
        labels = labels_sorted + labels
        styles = styles_sorted + STYLES
        colors = ["black", "black", "black"] + COLORS
        markers = ["", "", ""] + MARKERS
    else:
        paras_unsorted = [2, l]
        gaps_unsorted = [create_reference_line(ITERATIONS, gap_0, 2),
                         create_reference_line(ITERATIONS, gap_0, l)]
        labels_unsorted = [('gap' + r'$_0 \cdot t^{-2}$'),
                           ('gap' + r'$_0 \cdot t^{-\ell}$')]
        styles_unsorted = ["-.", ":"]
        sorted_lists = sorted(zip(paras_unsorted, gaps_unsorted, labels_unsorted, styles_unsorted))
        paras_sorted, gaps_sorted, labels_sorted, styles_sorted = zip(*sorted_lists)

        # Convert the sorted tuples back to lists
        paras_sorted = list(paras_sorted)
        gaps_sorted = list(gaps_sorted)
        labels_sorted = list(labels_sorted)
        styles_sorted = list(styles_sorted)

        gaps = gaps_sorted + gaps
        labels = labels_sorted + labels
        styles = styles_sorted + STYLES
        colors = ["black", "black"] + COLORS
        markers = ["", ""] + MARKERS

    file_name = ("strong_growth" + "_r=" + str(round(r, 2)) + "_M=" + str(round(M, 2)) + "_p=" + str(round(p, 2)) +
                 "_lmbda=" + str(round(lmbda, 2)))

    gap_plotter(y_data=gaps,
                labels=labels,
                iterations=ITERATIONS,
                file_name=("gaps_"+file_name),
                x_lim=(1, ITERATIONS),
                y_lim=determine_y_lims(primal_gaps),
                y_label=("Optimality measure"),
                directory="experiments/figures/strong_growth/",
                legend=True,
                styles=styles,
                colors=colors,
                markers=markers
                )




