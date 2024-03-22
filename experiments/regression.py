from sklearn.datasets import load_boston, fetch_california_housing

from src.feasible_region import LpBall, lpnorm
from src.objective_function import SquaredLoss
from src.experiments_auxiliary_functions import run_experiment, create_reference_lines_automatically
from global_ import *
import matplotlib as mpl
from src.plotting import gap_plotter, determine_y_lims
import random
import autograd.numpy as np
from sklearn.preprocessing import StandardScaler


random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH

for dataset in ["boston", "california"]:
    for location in ["interior", "boundary", "exterior"]:
        for p in[1., 2., 5.]:
            assert p >= 1, "Only consider lp-balls with p>= 1."
            if dataset == "boston":
                boston = load_boston()
                A = boston.data
                y = boston.target
            elif dataset == "california":
                california = fetch_california_housing()
                A = california.data
                y = california.target

            # Create a StandardScaler object for feature matrix and target variable separately
            scaler_A = StandardScaler()
            scaler_y = StandardScaler()

            # Perform Z-score normalization on feature matrix A
            A = scaler_A.fit_transform(A)
            m, n = A.shape
            print("Dimensions: ", (m, n))
            # Perform Z-score normalization on target variable y
            b = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

            objective_function = SquaredLoss(A=A, b=b, lmbda=0)
            optimum, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
            optimum_norm = lpnorm(optimum, p=p)

            if location == "interior":
                radius = optimum_norm * 3 / 2
            elif location == "boundary":
                radius = optimum_norm * 1
            elif location == "exterior":
                radius = optimum_norm * 1 / 2
            feasible_region = LpBall(dimension=A.shape[1], p=p, radius=radius)

            step_size_rules = []
            labels = []
            ls = [2, 4]
            for l in ls:
                step_size_rules.append({"step type": "open-loop", "a": l, "b": 1, "c": l, "d": 1})
                latex_string = r'$\eta_t = \frac{{{}}}{{{}}}$'.format(l, "t + " + str(l))
                labels.append(latex_string)
            step_size_rules = step_size_rules + [
                # {"step type": "line-search"},
                {"step type": "log"}
            ]
            labels = labels + [
                # "line-search",
                r'$\eta_t = \frac{2+\log(t+1)}{t+2+\log(t+1)}$'
            ]

            all_primal_gaps, all_dual_gaps, all_primal_dual_gaps, _ = run_experiment(ITERATIONS_MANY, objective_function,
                                                                                     feasible_region,
                                                                                     run_more=RUN_MORE_MANY,
                                                                                     fw_step_size_rules=step_size_rules)

            all_primal_gaps = [primal_gap[1:ITERATIONS_MANY] for primal_gap in all_primal_gaps]
            all_dual_gaps = [dual_gap[1:ITERATIONS_MANY] for dual_gap in all_dual_gaps]
            all_primal_dual_gaps = [primal_dual_gap[1:ITERATIONS_MANY] for primal_dual_gap in all_primal_dual_gaps]

            gap_0 = max([max(i) for i in all_dual_gaps])
            all_primal_gaps, labels, styles, colors, markers = create_reference_lines_automatically(
                all_primal_gaps, labels, None, None, gap_0, iterations=ITERATIONS_MANY, colors=COLORS)
            all_dual_gaps, _, _, _, _ = create_reference_lines_automatically(all_dual_gaps, labels, None, None, gap_0,
                                                                             iterations=ITERATIONS_MANY)
            all_primal_dual_gaps, _, _, _, _ = create_reference_lines_automatically(all_primal_dual_gaps, labels, None, None,
                                                                                    gap_0, iterations=ITERATIONS_MANY)
            file_name = (("regression" + "_" + str(dataset) + "_p=" + str(round(p, 2)) + "_location="+str(location)))

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