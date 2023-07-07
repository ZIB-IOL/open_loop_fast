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


ps = [1.01, 1.1, 2., 2.5, 3., 7.]
lmbdas = [0.2]
l = 4
for lmbda in lmbdas:
    for p in ps:
        L = 1.
        if p >= 2:
            alpha = 1/p
            M = L * (alpha*lmbda)**(-2/p)
            r = 2/p
            y = np.random.random((DIMENSION, 1))
            y = (lmbda + DIMENSION**(1/2 - 1/p)) * y / lpnorm(y, 2)
        elif p < 2:
            alpha = (p-1)/2
            M = L/(alpha*lmbda)
            r = 1.
            y = np.random.random((DIMENSION, 1))
            y = (lmbda + 1) * y / lpnorm(y, 2)


        A = np.identity(DIMENSION)


        objective_function = SquaredLoss(A=A, b=y)
        feasible_region = LpBall(dimension=DIMENSION, p=p)

        fw_step_size_rules = [{"step type": "open-loop", "a": l, "b": 1, "c": l, "d": 1}]
        primal_gaps, dual_gaps, best_gaps, _ = run_experiment(ITERATIONS, objective_function, feasible_region,
                                                                   run_more=RUN_MORE, fw_step_size_rules=fw_step_size_rules)

        gaps = [dual_gaps[0][1:ITERATIONS], best_gaps[0][1:ITERATIONS], primal_gaps[0][1:ITERATIONS]]
        labels = ["gap" + r'$_t$', "bestgap" + r'$_t$', "subopt" + r'$_t$']
        gap_0 = dual_gaps[0][0]

        if r < 1:
            paras_unsorted = [1 /(1-r), l]
            gaps_unsorted = [create_reference_line(ITERATIONS, gap_0, 1 / (1 - r)),
                             create_reference_line(ITERATIONS, gap_0, l)]
            labels_unsorted = [(r'$ \mathcal{O} (t^{-\frac{1}{1-r}})$'), (r'$\mathcal{O} (t^{-\ell})$')]
            styles_unsorted = ["--", ":"]
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
        else:
            paras_unsorted = [l]
            gaps_unsorted = [create_reference_line(ITERATIONS, gap_0, l)]
            labels_unsorted = [(r'$\mathcal{O} (t^{-\ell})$')]
            styles_unsorted = [":"]
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
            colors = ["black"] + COLORS
            markers = [""] + MARKERS

        file_name = ("strong_growth" + "_r=" + str(round(r, 2)) + "_M=" + str(round(M, 2)) + "_p=" + str(round(p, 2)) +
                     "_lmbda=" + str(round(lmbda, 2)) + "_l=" + str(l))

        S = max(np.int64(np.ceil(M*l/2 - l)), 0)
        S_label = "S = " + str(S)
        lines = [(S, S_label)]

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
                    markers=markers,
                    vertical_lines=lines
                    )




