# Logistic regression experiment over different lp-balls using the gisette dataset.

import random
import autograd.numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.feasible_region import LpBall, lpnorm
from src.frank_wolfe import frank_wolfe
from src.objective_function import SquaredLoss
from src.experiments_auxiliary_functions import run_experiment, create_reference_line
from global_ import *
import matplotlib as mpl
from sklearn.datasets import load_boston

import sympy as sp

from src.plotting import gap_plotter, determine_y_lims

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH

# TODO: normalization discussion
print("todo: write about normalization")
boston = load_boston()
A = boston.data
y = boston.target
# Create a StandardScaler object for feature matrix and target variable separately
scaler_A = StandardScaler()
scaler_y = StandardScaler()
# Perform Z-score normalization on feature matrix A
A = scaler_A.fit_transform(A)
# Perform Z-score normalization on target variable y
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

x, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)

objective_function = SquaredLoss(A=A, b=y, lmbda=0)
optimum, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)

L = objective_function.L
min_eigenvalue = objective_function.min_eigenvalue
theta = 1 / 2
mu = (2 / min_eigenvalue) ** theta

l = 4
lmbda = 0.2

ps = [1.1, 1.5, 2., 2.5, 3., 5., 7.]
locations = ["interior", "border", "exterior"]

locations = ["border"]

for p in ps:
    for location in locations:
        if location == "interior":
            print("IMPLEMENT")
        elif location == "border":
            radius = lpnorm(optimum, p)
        elif location == "exterior":
            print("IMPLEMENT!")

        feasible_region = LpBall(dimension=A.shape[1], p=p, radius=radius)

        if p >= 2:
            alpha = 1 / p
            diameter = 2 * DIMENSION ** (1 / 2 - 1 / p)
            M = L * (mu / alpha) ** (2 / p)
            r = 2 * theta / p

        elif p < 2:
            alpha = (p - 1) / 2
            diameter = 2
            M = L * mu / alpha
            r = theta

        # incorporates the strong (M, 0)-growth
        M = max(M, diameter ** 2 * L)

        # if p >= 2:
        #     alpha = 1/p
        #     M = L*(alpha*lmbda)**(-2/p)
        #     r = 2/p
        #
        # elif p < 2:
        #     alpha = (p-1)/2
        #     M = L/(alpha*lmbda)
        #     r = 1.
        #     y = np.random.random((DIMENSION, 1))
        #     y = (lmbda + 1) * y / lpnorm(y, 2)

        fw_step_size_rules = [{"step type": "open-loop", "a": l, "b": 1, "c": l, "d": 1}]
        primal_gaps, dual_gaps, best_gaps, _ = run_experiment(ITERATIONS, objective_function, feasible_region,
                                                              run_more=RUN_MORE,
                                                              fw_step_size_rules=fw_step_size_rules)

        gaps = [dual_gaps[0][1:ITERATIONS], best_gaps[0][1:ITERATIONS], primal_gaps[0][1:ITERATIONS]]
        labels = ["gap" + r'$_t$', "bestgap" + r'$_t$', "subopt" + r'$_t$']
        gap_0 = dual_gaps[0][0]

        if r < 1 and r != 0.5:
            paras_unsorted = [1 / (1 - r), 2, l]
            gaps_unsorted = [create_reference_line(ITERATIONS, gap_0, 1 / (1 - r)),
                             create_reference_line(ITERATIONS, gap_0, 2),
                             create_reference_line(ITERATIONS, gap_0, l)]
            # labels_unsorted = [('gap' + r'$_0 \cdot t^{-\frac{1}{1-r}}$'),
            #                    ('gap' + r'$_0 \cdot t^{-2}$'),
            #                    ('gap' + r'$_0 \cdot t^{-\ell}$')]
            labels_unsorted = [(r'$ \mathcal{O} (t^{-\frac{1}{1-r}})$'), (r'$\mathcal{O} ( t^{-2})$'),
                               (r'$\mathcal{O} (t^{-\ell})$')]
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
            paras_unsorted = [1 / (1 - r), l]
            gaps_unsorted = [create_reference_line(ITERATIONS, gap_0, 1 / (1 - r)),
                             create_reference_line(ITERATIONS, gap_0, l)]
            # labels_unsorted = [('gap' + r'$_0 \cdot t^{-2}$'),
            #                    ('gap' + r'$_0 \cdot t^{-\ell}$')]
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

        file_name = ("regression" + "_location=" + str(location) + "_r=" + str(round(r, 2)) + "_M=" + str(round(M, 2))
                     + "_p=" + str(round(p, 2)))

        S = max(np.int64(np.ceil(M * l / 2 - l)), 0)
        print("S : ", S)
        S_label = "S = " + str(S)
        lines = [(S, S_label)]

        gap_plotter(y_data=gaps,
                    labels=labels,
                    iterations=ITERATIONS,
                    file_name=("gaps_" + file_name),
                    x_lim=(1, ITERATIONS),
                    y_lim=determine_y_lims(primal_gaps),
                    y_label=("Optimality measure"),
                    directory="experiments/figures/regression/",
                    legend=True,
                    styles=styles,
                    colors=colors,
                    markers=markers,
                    vertical_lines=lines
                    )
