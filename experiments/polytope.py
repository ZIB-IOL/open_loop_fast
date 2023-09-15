import random
import autograd.numpy as np
from src.feasible_region import lpnorm, LpBall
from src.objective_function import SquaredLoss
from src.plotting import gap_plotter, determine_y_lims
from src.experiments_auxiliary_functions import run_experiment, create_reference_lines_automatically
from global_ import *
import matplotlib as mpl

random.seed(RANDOM)
np.random.seed(RANDOM)

mpl.rcParams['agg.path.chunksize'] = CHUNKSIZE
mpl.rcParams['axes.linewidth'] = LINEWIDTH


rhos = [0.1]

kappas = [0.0001, 0.01, 100]
l = 4
for kappa in kappas:
    for rho in rhos:
        z = np.random.random((DIMENSION, 1))
        z[0] = 0
        z[1] = 0
        z = rho * z / lpnorm(z, 1)
        z[1] = (1 - rho)
        offset = kappa * np.ones((DIMENSION, 1))
        offset[0] = 0
        y = z + offset
        # Note that x* = z
        L = 1.
        mu = np.sqrt(2) * np.sqrt(DIMENSION)
        theta = 1 / 2
        r = theta
        m = rho /((2**(2-theta))*mu)
        Q = np.ceil(4*L*l * ((4*L*mu/kappa)**(1/theta)))
        eta_Q = l/(Q+l)
        B = 2 - rho
        M = max(m/eta_Q*((4*B)**(1-theta)), 2*(2*L+B)/eta_Q)

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
        labels = ["gap" + r'$_t$', "primaldual" + r'$_t$', "subopt" + r'$_t$']
        gap_0 = dual_gaps[0][0]
        gaps, labels, styles, colors, markers = create_reference_lines_automatically(gaps, labels, 1, l, gap_0)
        file_name = ("polytope_growth" + "_l1_ball" + "_rho=" + str(rho) + "_kappa=" + str(kappa) + "_l=" + str(l))

        # compute S
        # S = int(max(1, np.ceil(2**(1/r)*l*M/(m**(1/r)) - l), Q))
        # S_label = "S = " + str(S)
        # lines = [(S, S_label)]

        # Q_label = "Q = " + str(Q)
        # lines = [(Q, Q_label)]

        lines = None

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
