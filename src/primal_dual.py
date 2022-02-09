from typing import List

import numpy as np
import time
import src.optimization as optimization

from src import const
from src.data.portfolio_data import random_portfolios
from src.entities import Portfolio, PrimalDualParameters


"""
Structure of the data:
    FOr the Primal dual algorithm, the portfolio is thought as a probability vector in a grid of possible investiment classes. 
    These classes are in the cartesian set R x {group}

    The portfolio is represented by a class Portfolio composed by elements p, esg, creditspread, sector. 
    For each time t, portfolios[t].get_p_list() is a probability vector, where t in {0,...,T}.
    portfolios[t].get_p_list() is a vector such that 0 < portfolios[t].get_p_list()[i] < 1 and sum(portfolios[t].get_p_list()) = 1 for each t. 
    portfolios[t].get_p(i,j) represents the percentage of investiments in an investiment class {portfolios[t].get_esg(i,j).portfolios[t].get_sector(i,j)}, an element in R2 (sectors correspond to numbers)
    i,j in {0,...,N} represents the possible investiment classes.

    The primal dual algorithm works directly with the vector portfolios[t].get_p_list(). Other elements (esg,sector,creditspread) do not change 
    in the execution of the code, reason why we use auxiliars z_next, z_a that are only the probability vectors.
    the components {portfolios[t].get_esg(),portfolios[t].get_credit_spread()} are used only to provide the transport cost, used in the function bilinear_max/cost_matrix

    # T number of time steps
    # tau_fixed constant of the primal dual algorithm
    # N = granularity of the grid
    # lambda_vec = wieght vector. Suggestion: lambda_vec = [1/(T+2)]*(T+4)


"""


def run_primal_dual_test(params: PrimalDualParameters, p_initial: Portfolio, p_ref: Portfolio, **kwargs):
    # Processing flags
    f_enable_random = kwargs.get('enable_random', True)
    f_verbose = kwargs.get('verbose', False)

    T, cte, tau, min_gap, max_iter, lambda_vec = params.as_tuple()

    sigma = 3 / (2 * tau)  # sigma as a proportion of tau, suggestion:sigma = 1/200; tau = 200;
    n = len(p_ref.get_p_list())
    N = p_initial.get_len()

    inter_dist = T + 1  # quantity of intermediate time step

    # Initial transition distributions
    intermediary_portfolios = random_portfolios(p_ref, T + 1, f_enable_random)
    portfolios = [p_initial] + intermediary_portfolios + [p_ref]

    # risk vector computed according to definition in "Measuring Risk" section and the linearity of VaR
    risk_reference = const.risk_reference

    # Selection of risk for portfolio actity sectors
    sector = list(set(p_initial.get_unique_sector_list()))
    const_risk = 10  # constant to normalize risk in the objective function
    risk = [const_risk * risk_reference[sector[i]] for i in range(len(sector))] * N

    # find initial f and g
    f, g = _initialization(portfolios, cte)

    # initialization
    W = []
    Inf = []

    z_initial = p_initial.get_p_list()
    z_ref = p_ref.get_p_list()

    f_next = []
    g_next = []
    z_next = []
    f_next += f
    g_next += g
    z_next += [port.get_p_list() for port in portfolios]

    # primal dual algorithm
    for k in range(max_iter):
        start_time = time.time()

        f_a = []
        g_a = []
        z_a = []
        f_a += f_next
        g_a += g_next
        z_a += z_next

        # optimization - mininimization
        z_hat = []
        z_hat = z_hat + [z_initial]
        for t in range(1, inter_dist + 1):
            z_t = [z_a[t][i] - sigma * (lambda_vec[t] * (f_a[t][i]) + lambda_vec[t - 1] * (g_a[t][i])) for i in
                   range(n)]
            z_hat = z_hat + [z_t]
        z_hat = z_hat + [z_ref]

        z_next = []
        z_next = z_next + [z_initial]

        for t in range(1, inter_dist + 1):
            z_next_t = optimization.bilinear_min_obj(z_hat, risk, t, sigma)  # min part of the optimization problem
            for i in range(n):  # (small values -> zero) to avoid numerical problems
                if z_next_t[i] < 1e-07:
                    z_next_t[i] = 0
            z_next_t = [z_next_t[i] / sum(z_next_t) for i in range(n)]
            z_next = z_next + [z_next_t]
        z_next = z_next + [z_ref]

        # defining z_bar
        z_bar = []
        z_bar = z_bar + [z_initial]
        for t in range(1, inter_dist + 1):
            z_bar_t = [[2 * z_next[t][i] - z_a[t][i] for i in range(n)]]
            z_bar = z_bar + z_bar_t
        z_bar = z_bar + [z_ref]

        # updating values of f and g
        f_hat = []
        g_hat = [[0]]
        f_hat += f_hat + [[f_a[0][i] + tau * lambda_vec[0] * z_initial[i] for i in range(len(z_initial))]]

        for t in range(1, T + 2):
            f_t = [[f_a[t][i] + tau * (lambda_vec[t] * z_bar[t][i]) for i in range(n)]]
            g_t = [[g_a[t][i] + tau * (lambda_vec[t - 1] * z_bar[t][i]) for i in range(n)]]
            f_hat = f_hat + f_t
            g_hat = g_hat + g_t
        g_hat = g_hat + [[g_a[T + 2][i] + tau * lambda_vec[T + 1] * z_ref[i] for i in range(len(z_ref))]]

        # optimization - maximazation part
        f_next = []
        g_next = [[0] * (N ** 2)]
        for t in range(inter_dist + 1):
            f_next_t, g_next_t = optimization.bilinear_max(f_hat, g_hat, t, tau, portfolios[t], portfolios[t + 1], cte)
            f_next = f_next + [f_next_t]
            g_next = g_next + [g_next_t]
        f_next = f_next + [[0] * (N ** 2)]

        for i, portfolio in enumerate(portfolios):
            for j, v in enumerate(portfolio.flatten()):
                v.p = z_next[i][j]
        Inf, W, inf, w = optimization.gap_primal_dual(W, Inf, T, risk, lambda_vec, portfolios, f_next, g_next, cte)
        gap_size = np.abs(w - inf)

        execution_time = (time.time() - start_time)
        if f_verbose:
            print("Iteration {0} finished. time elapsed: {1}. Current Gap: {2}".format(str(k+1), execution_time, gap_size))

        # test the primal-dual gap at each 3 iterations
        final_iteration = (k == (max_iter-1))

        if k % 3 == 0:
            gap_small = gap_size < min_gap
            if final_iteration or gap_small:
                return portfolios, execution_time, W, Inf, risk
        else:
            Inf = Inf + [inf]
            W = W + [w]

    return None, None, None, None, None


def _initialization(portfolios: List[Portfolio], cte: float) -> (List[float], List[float]):
    f = []
    g = [[0]]

    for i in range(len(portfolios) - 1):
        _, f_aux, g_aux = optimization.wasserstein(portfolios[i], portfolios[i + 1], cte)
        f.extend([f_aux])  # the initial f and g are the same for every time step
        g.extend([g_aux])

    return f, g
