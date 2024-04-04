
from src.feasible_region import NuclearNormBall
from src.objective_function import HuberLossCollaborativeFiltering
from src.experiments_auxiliary_functions import run_experiment, create_reference_lines_automatically
from global_ import *
import matplotlib as mpl
from src.plotting import gap_plotter, determine_y_lims
import os
import random
import autograd.numpy as np
import pandas as pd

import pickle


random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH

radii = [2000]
for radius in radii:
    data = pd.read_csv(os.path.dirname(__file__) + '/../datasets/movielens100k.csv',
                       names=['user id', 'item id', 'rating', 'timestamp'])
    A = pd.pivot_table(data, values='rating', index='user id', columns='item id').values
    m, n = A.shape

    objective_function = HuberLossCollaborativeFiltering(A=A)
    feasible_region = NuclearNormBall(m, n, radius=radius)

    print("Dimensions: ", (m, n))

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

    all_primal_gaps, all_dual_gaps, all_primal_dual_gaps, _ = run_experiment(ITERATIONS_FEW, objective_function,
                                                                             feasible_region,
                                                                             run_more=RUN_MORE_FEW,
                                                                             fw_step_size_rules=step_size_rules)
    all_primal_gaps = [primal_gap[1:ITERATIONS_FEW] for primal_gap in all_primal_gaps]
    all_dual_gaps = [dual_gap[1:ITERATIONS_FEW] for dual_gap in all_dual_gaps]
    all_primal_dual_gaps = [primal_dual_gap[1:ITERATIONS_FEW] for primal_dual_gap in all_primal_dual_gaps]

    gap_0 = max([max(i) for i in all_dual_gaps])
    all_primal_gaps, labels, styles, colors, markers = create_reference_lines_automatically(
        all_primal_gaps, labels, None, None, gap_0, iterations=ITERATIONS_FEW, colors=COLORS)
    all_dual_gaps, _, _, _, _ = create_reference_lines_automatically(all_dual_gaps, labels, None, None, gap_0,
                                                                     iterations=ITERATIONS_FEW)
    all_primal_dual_gaps, _, _, _, _ = create_reference_lines_automatically(all_primal_dual_gaps, labels, None, None,
                                                                            gap_0, iterations=ITERATIONS_FEW)
    file_name = ("collaborative_filtering" +  "_radius=" + str(radius))

    # Prepare the data to be saved, now including styles, colors, and markers
    data_to_save = {
        "file_name": file_name,
        "all_primal_gaps": all_primal_gaps,
        "all_dual_gaps": all_dual_gaps,
        "all_primal_dual_gaps": all_primal_dual_gaps,
        "labels": labels,
        "styles": styles,  # Assuming you have a variable styles defined
        "colors": colors,  # Assuming you have a variable colors defined
        "markers": markers  # Assuming you have a variable markers defined
    }

    # Specify the directory and file path
    directory_path = "experiments/data/"
    file_path = f"{directory_path}{file_name}.pkl"

    # Check if the directory exists, and create it if it does not
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Write the serialized data to file
    with open(file_path, 'wb') as file:
        pickle.dump(data_to_save, file)

    # # Load the serialized data from file
    # with open(file_path, 'rb') as file:
    #     data_loaded = pickle.load(file)
    #
    # # Extract the loaded data
    # file_name = data_loaded["file_name"]
    # all_primal_gaps = data_loaded["all_primal_gaps"]
    # all_dual_gaps = data_loaded["all_dual_gaps"]
    # all_primal_dual_gaps = data_loaded["all_primal_dual_gaps"]
    # labels = data_loaded["labels"]
    # styles = data_loaded["styles"]
    # colors = data_loaded["colors"]
    # markers = data_loaded["markers"]
    #
    # y_label = "subopt" + r"$_t$"
    # gap_plotter(y_data=all_primal_gaps,
    #             labels=labels,
    #             iterations=ITERATIONS_FEW,
    #             file_name=("subopt_" + file_name),
    #             x_lim=(1, ITERATIONS_FEW),
    #             y_lim=determine_y_lims(all_primal_gaps),
    #             y_label=y_label,
    #             directory="experiments/figures/",
    #             legend=True,
    #             styles=styles,
    #             colors=colors,
    #             markers=markers
    #             )
    #
    # y_label = "gap" + r"$_t$"
    # gap_plotter(y_data=all_dual_gaps,
    #             labels=labels,
    #             iterations=ITERATIONS_FEW,
    #             file_name=("gap_" + file_name),
    #             x_lim=(1, ITERATIONS_FEW),
    #             y_lim=determine_y_lims(all_dual_gaps),
    #             y_label=y_label,
    #             directory="experiments/figures/",
    #             legend=True,
    #             styles=styles,
    #             colors=colors,
    #             markers=markers
    #             )
    #
    # y_label = "primaldual" + r"$_t$"
    # gap_plotter(y_data=all_primal_dual_gaps,
    #             labels=labels,
    #             iterations=ITERATIONS_FEW,
    #             file_name=("primaldual_" + file_name),
    #             x_lim=(1, ITERATIONS_FEW),
    #             y_lim=determine_y_lims(all_primal_dual_gaps),
    #             y_label=y_label,
    #             directory="experiments/figures/",
    #             legend=True,
    #             styles=styles,
    #             colors=colors,
    #             markers=markers
    #             )