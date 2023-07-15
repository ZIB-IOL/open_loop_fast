# Logistic regression experiment over different lp-balls using the gisette dataset.

import random
import autograd.numpy as np
from sklearn.preprocessing import StandardScaler
from src.feasible_region import LpBall, lpnorm
from src.objective_function import SquaredLoss
from src.experiments_auxiliary_functions import run_experiment, create_reference_lines_automatically
from global_ import *
import matplotlib as mpl
from sklearn.datasets import load_boston
from src.plotting import gap_plotter, determine_y_lims

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH

ps = [1.0, 2.0]
locations = ["interior", "border", "exterior"]

l = 4
for p in ps:
    boston = load_boston()
    A = boston.data
    y = boston.target

    # Create a StandardScaler object for feature matrix and target variable separately
    scaler_A = StandardScaler()
    scaler_y = StandardScaler()

    # Perform Z-score normalization on feature matrix A
    A = scaler_A.fit_transform(A)
    print("A.shape: ", A.shape)
    # Perform Z-score normalization on target variable y
    b = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    objective_function = SquaredLoss(A=A, b=b, lmbda=0)
    optimum, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    optimum_norm = lpnorm(optimum, p=p)

    all_primal_gaps = []
    all_dual_gaps = []
    all_best_gaps = []
    labels = []
    for location in locations:
        if location == "interior":
            radius = optimum_norm * 3/2
        elif location == "border":
            radius = optimum_norm * 1
        elif location == "exterior":
            radius = optimum_norm * 1/2
        feasible_region = LpBall(dimension=A.shape[1], p=p, radius=radius)
        fw_step_size_rules = [{"step type": "open-loop", "a": l, "b": 1, "c": l, "d": 1}]
        primal_gaps, dual_gaps, best_gaps, _ = run_experiment(ITERATIONS, objective_function, feasible_region,
                                                              run_more=RUN_MORE,
                                                              fw_step_size_rules=fw_step_size_rules)
        labels.append(location)
        all_primal_gaps.append(primal_gaps[0][1:ITERATIONS])
        all_dual_gaps.append(dual_gaps[0][1:ITERATIONS])
        all_best_gaps.append(best_gaps[0][1:ITERATIONS])

    gap_0 = max([max(i) for i in all_dual_gaps])
    all_primal_gaps, labels, styles, colors, markers = create_reference_lines_automatically(all_primal_gaps,
                                                                                            labels, 1, l, gap_0)
    all_dual_gaps, _, _, _, _ = create_reference_lines_automatically(all_dual_gaps, labels, 1, l, gap_0)
    all_best_gaps, _, _, _, _ = create_reference_lines_automatically(all_best_gaps, labels, 1, l, gap_0)
    file_name = ("regression" + "_p=" + str(round(p, 2)) + "_l=" + str(l))



    y_label = "subopt"
    y_label = "subopt" + r"$_t$"
    gap_plotter(y_data=all_primal_gaps,
                labels=labels,
                iterations=ITERATIONS,
                file_name=("subopt_" + file_name),
                x_lim=(1, ITERATIONS),
                y_lim=determine_y_lims(primal_gaps),
                y_label=y_label,
                directory="experiments/figures/regression/",
                legend=True,
                styles=styles,
                colors=colors,
                markers=markers
                )

    y_label = "gap"
    y_label = "gap" + r"$_t$"
    gap_plotter(y_data=all_dual_gaps,
                labels=labels,
                iterations=ITERATIONS,
                file_name=("gap_" + file_name),
                x_lim=(1, ITERATIONS),
                y_lim=determine_y_lims(primal_gaps),
                y_label=y_label,
                directory="experiments/figures/regression/",
                legend=True,
                styles=styles,
                colors=colors,
                markers=markers
                )

    y_label = "bestgap"
    y_label = "bestgap" + r"$_t$"
    gap_plotter(y_data=all_best_gaps,
                labels=labels,
                iterations=ITERATIONS,
                file_name=("bestgap_" + file_name),
                x_lim=(1, ITERATIONS),
                y_lim=determine_y_lims(primal_gaps),
                y_label=y_label,
                directory="experiments/figures/regression/",
                legend=True,
                styles=styles,
                colors=colors,
                markers=markers
                )
