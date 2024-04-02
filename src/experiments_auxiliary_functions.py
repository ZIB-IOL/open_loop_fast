from global_ import *
from src.frank_wolfe import frank_wolfe, decomposition_invariant_frank_wolfe, away_step_frank_wolfe, \
    momentum_guided_frank_wolfe, primal_averaging_frank_wolfe
import autograd.numpy as np
from scipy import stats


def primal_dual_computer(primal_gaps, dual_gaps):
    """
    From primal_gaps and dual_gaps, computes primal_dual_gaps.
    """
    # val_smallest stores min_{k <= t} -f(x_k) + gap_k
    val_smallest = 10.0**16
    primal_dual_gaps = []
    for i in range(0, len(primal_gaps)):
        val_smallest = min(val_smallest, - primal_gaps[i] + dual_gaps[i])
        primal_dual_gaps.append(primal_gaps[i] + val_smallest)
    return primal_dual_gaps


def create_reference_line(iterations, constant, exponent):
    """
    Creates a list of values of length iterations such that the ith value is constant/i^exponent.
    """
    ref_line = [constant]
    for iteration in range(1, iterations+1):
        ref_line.append(constant/iteration**exponent)
    return ref_line



def create_reference_lines_automatically(gaps, labels, r, l, c, iterations=ITERATIONS, styles=STYLES, colors=COLORS,
                                         markers=MARKERS):
    """
    Adds the reference convergence lines and all corresponding parameters such that everything is plotting-ready.
    """
    if r == None and l == None:
        paras_unsorted = [2]
        gaps_unsorted = [create_reference_line(iterations, c, 2)]
        labels_unsorted = [(r'$\mathcal{O} ( t^{-2})$')]
        styles_unsorted = ["-."]
        colors = ["black"] + colors
        markers = [""] + markers

    elif r == 1/2:
        paras_unsorted = [1 / (1 - r), l]
        gaps_unsorted = [create_reference_line(ITERATIONS, c, 1 / (1 - r)),
                         create_reference_line(ITERATIONS, c, l)]
        labels_unsorted = [(r'$ \mathcal{O} (t^{-\frac{1}{1-r}})$'),
                           (r'$\mathcal{O} (t^{-\ell})$')]
        styles_unsorted = ["--", ":"]
        colors = ["black", "black"] + colors
        markers = ["", ""] + markers

    elif r == 1:
        paras_unsorted = [2, l]
        gaps_unsorted = [create_reference_line(iterations, c, 2),
                         create_reference_line(iterations, c, l)]
        labels_unsorted = [(r'$\mathcal{O} ( t^{-2})$'), (r'$\mathcal{O} (t^{-\ell})$')]
        styles_unsorted = ["-.", ":"]
        colors = ["black", "black"] + colors
        markers = ["", ""] + markers

    else:
        paras_unsorted = [1 / (1 - r), 2, l]
        gaps_unsorted = [create_reference_line(iterations, c, 1 / (1 - r)),
                         create_reference_line(iterations, c, 2),
                         create_reference_line(iterations, c, l)]
        labels_unsorted = [(r'$ \mathcal{O} (t^{-\frac{1}{1-r}})$'), (r'$\mathcal{O} ( t^{-2})$'),
                           (r'$\mathcal{O} (t^{-\ell})$')]
        styles_unsorted = ["--", "-.", ":"]
        colors = ["black", "black", "black"] + colors
        markers = ["", "", ""] + markers


    sorted_lists = sorted(zip(paras_unsorted, gaps_unsorted, labels_unsorted, styles_unsorted))
    paras_sorted, gaps_sorted, labels_sorted, styles_sorted = zip(*sorted_lists)

    # Convert the sorted tuples back to lists
    gaps_sorted = list(gaps_sorted)
    labels_sorted = list(labels_sorted)
    styles_sorted = list(styles_sorted)

    gaps = gaps_sorted + gaps
    labels = labels_sorted + labels
    styles = styles_sorted + styles


    return gaps, labels, styles, colors, markers


def compute_convergence_rates(data, n_iterates):
    """Computes the convergence rate of the data according to the order estimation procedure in [1]

    References:
        [1] Senning, Jonathan R. "Computing and Estimating the Rate of Convergence" (PDF). gordon.edu.
        Retrieved 2021-05-18.
    """
    data = np.log(data)
    xdata = np.asarray(list(np.log(range(1, len(data) + 1))))
    convergence_rates = [-stats.linregress(xdata[i:i + n_iterates], data[i:i + n_iterates])[0] for i in
                         range(0, len(data) - n_iterates)]

    return convergence_rates


def translate_step_types(current_label, step):
    """
    Translates the step types.
    """
    if step["step type"] == "open-loop":
        current_label = current_label + " " + "open-loop, " + " " + r"$\ell={}$".format(str(int(step["a"])))
    if step["step type"] == "open-loop constant":
        current_label = current_label + " " + "constant"
    elif step["step type"] == "log":
        current_label = current_label + " " + "log"
    elif step["step type"] in ["line-search", "line-search difw probability simplex", "line-search afw"]:
        current_label = current_label + " " + "line-search"
    elif step["step type"] in ["short-step", "short-step difw probability simplex", "short-step afw"]:
        current_label = current_label + " " + "short-step"
    return current_label


def run_experiment(iterations,
                   objective_function,
                   feasible_region,
                   run_more: int = 0,
                   fw_step_size_rules: list = [],
                   difw_step_size_rules: list = [],
                   afw_step_size_rules: list = [],
                   mfw_step_size_rules: list = [],
                   pafw_step_size_rules: list = []
                   ):
    """
    Minimizes objective_function over the feasible_region.

    Args:
        iterations: int
            The number of iterations.
        objective_function
        feasible_region
        run_more: int, Optional
            Number of additional ITERATIONS to run to determine f(x*). (Default is 0.)
        fw_step_size_rules: list
            The types of FW step-size rules we want to run. (Default is [].)
        difw_step_size_rules: list
            The types of DIFW step-size rules we want to run. (Default is [].)
        afw_step_size_rules: list
            The types of AFW step-size rules we want to run. (Default is [].)
        mfw_step_size_rules: list
            The types of MFW step-size rules we want to run. (Default is [].)
        pafw_step_size_rules: list
            The types of PAFW step-size rules we want to run. (Default is [].)

    Returns:
        Returns primal_data, dual_data, primal_dual_data, labels.
    """
    labels = []
    primal_data = []
    dual_data = []
    primal_dual_data = []
    for step in fw_step_size_rules:
        print(".")
        current_label = translate_step_types("FW", step)

        iterate_list, loss_list, fw_gap_list, x, x_p_list = frank_wolfe(feasible_region=feasible_region,
                                                                        objective_function=objective_function,
                                                                        n_iters=(int(iterations + run_more)),
                                                                        step=step)

        dual_data_list = fw_gap_list
        primal_dual_data_list = primal_dual_computer(loss_list, fw_gap_list)
        if run_more == 0:
            primal_data_list = loss_list
        else:
            primal_data_list = [loss_list[i] - loss_list[-1] for i in range(len(loss_list))][:iterations]
        primal_data.append(primal_data_list)
        dual_data.append(dual_data_list)
        primal_dual_data.append(primal_dual_data_list)
        labels.append(current_label)
    for step in afw_step_size_rules:
        current_label = translate_step_types("AFW", step)
        iterate_list, loss_list, fw_gap_list, x, x_p_list = away_step_frank_wolfe(feasible_region=feasible_region,
                                                                                  objective_function=objective_function,
                                                                                  n_iters=(int(iterations + run_more)),
                                                                                  step=step)
        dual_data_list = fw_gap_list
        primal_dual_data_list = primal_dual_computer(loss_list, fw_gap_list)
        if run_more == 0:
            primal_data_list = loss_list
        else:
            primal_data_list = [loss_list[i] - loss_list[-1] for i in range(len(loss_list))][:iterations]
        primal_data.append(primal_data_list)
        dual_data.append(dual_data_list)
        primal_dual_data.append(primal_dual_data_list)
        labels.append(current_label)
    for step in difw_step_size_rules:
        current_label = translate_step_types("DIFW", step)
        iterate_list, loss_list, fw_gap_list, x, x_p_list = decomposition_invariant_frank_wolfe(
            feasible_region=feasible_region,
            objective_function=objective_function,
            n_iters=(int(iterations + run_more)),
            step=step)
        dual_data_list = fw_gap_list
        primal_dual_data_list = primal_dual_computer(loss_list, fw_gap_list)
        if run_more == 0:
            primal_data_list = loss_list
        else:
            primal_data_list = [loss_list[i] - loss_list[-1] for i in range(len(loss_list))][:iterations]
        primal_data.append(primal_data_list)
        dual_data.append(dual_data_list)
        primal_dual_data.append(primal_dual_data_list)
        labels.append(current_label)
    for step in mfw_step_size_rules:
        current_label = translate_step_types("MFW", step)
        iterate_list, loss_list, fw_gap_list, x, x_p_list = momentum_guided_frank_wolfe(
            feasible_region=feasible_region, objective_function=objective_function,
            n_iters=(int(iterations + run_more)), step=step)

        dual_data_list = fw_gap_list
        primal_dual_data_list = primal_dual_computer(loss_list, fw_gap_list)
        if run_more == 0:
            primal_data_list = loss_list
        else:
            primal_data_list = [loss_list[i] - loss_list[-1] for i in range(len(loss_list))][:iterations]
        primal_data.append(primal_data_list)
        dual_data.append(dual_data_list)
        primal_dual_data.append(primal_dual_data_list)
        labels.append(current_label)

    for step in pafw_step_size_rules:
        current_label = translate_step_types("PAFW", step)
        iterate_list, loss_list, fw_gap_list, x, x_p_list = primal_averaging_frank_wolfe(
            feasible_region=feasible_region, objective_function=objective_function,
            n_iters=(int(iterations + run_more)), step=step)

        dual_data_list = fw_gap_list
        primal_dual_data_list = primal_dual_computer(loss_list, fw_gap_list)
        if run_more == 0:
            primal_data_list = loss_list
        else:
            primal_data_list = [loss_list[i] - loss_list[-1] for i in range(len(loss_list))][:iterations]
        primal_data.append(primal_data_list)
        dual_data.append(dual_data_list)
        primal_dual_data.append(primal_dual_data_list)
        labels.append(current_label)

    return primal_data, dual_data, primal_dual_data, labels




