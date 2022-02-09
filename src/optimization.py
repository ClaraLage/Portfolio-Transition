from typing import List

import numpy as np
import warnings
from cvxopt import solvers
from cvxopt import matrix
from scipy.optimize import linprog

from src.entities import Portfolio


def wasserstein(p1: Portfolio, p2: Portfolio, cte: float) -> (float, List[float], List[float]):
    alpha = p1.get_p_list()
    beta = p2.get_p_list()
    c = []
    c.extend(alpha)
    c.extend(beta)
    c = [-x for x in c]

    A, b = cost_matrix(p1, p2, cte)
    w1 = linprog(c, A, b, A_eq=None, b_eq=None, bounds=(None, None))

    return -w1.fun, [w1.x[i] for i in range(len(alpha))], [w1.x[i] for i in range(len(alpha), len(alpha) + len(beta))]


def cost_matrix(p1: Portfolio, p2: Portfolio, cte: float):
    n = p1.get_len()**2
    m = p2.get_len()**2

    b = []
    A = []
    for i, v_z1 in enumerate(p1.flatten()):
        for j, v_z2 in enumerate(p2.flatten()):
            zeros_alpha_1 = [0]*i
            zeros_alpha_2 = [0]*(n-i-1)
            A_alpha = zeros_alpha_1
            A_alpha.extend([1])
            A_alpha.extend(zeros_alpha_2)
            zeros_beta_1 = [0]*j
            zeros_beta_2 = [0]*(m-j-1)
            A_beta = zeros_beta_1
            A_beta.extend([1])
            A_beta.extend(zeros_beta_2)
            A_aux = A_alpha
            A_aux.extend(A_beta)
            A.append(A_aux)

            b.extend([abs(v_z1.esg - v_z2.esg) + cte*abs(v_z2.credit_spread - v_z1.credit_spread)])
    return A, b


def bilinear_max(f_hat, g_hat, t: int, tau: float, p1: Portfolio, p2: Portfolio, const: float):
    n = len(f_hat[t])
    m = len(g_hat[t + 1])
    q_aux = [-(1 / tau) * (f_hat[t] + g_hat[t + 1])[i] for i in range(n + m)]  # A^Ty_k
    Id = np.identity(n + m)
    Q_aux = [[1 / tau * Id[i][j] for i in range(n + m)] for j in range(n + m)]
    A_aux, b_aux = cost_matrix(p1, p2, const)
    A_g = [[0] for i in range(n)] + [[1] for i in range(m)]
    b_g = [0]

    Q = matrix(Q_aux, tc='d')
    q = matrix(q_aux, tc='d')
    G = matrix(np.transpose(A_aux).tolist(), tc='d')
    h = matrix(b_aux, tc='d')
    A = matrix(A_g, tc='d')
    b = matrix(b_g, tc='d')

    solvers.options['show_progress'] = False
    fg_next = solvers.qp(Q, q, G, h, A, b)
    fg_solution = [fg_next['x'][i] for i in range(n)], [fg_next['x'][i] for i in range(n, n + m)]
 
    return fg_solution


def bilinear_min(z_hat, risk, t: int, sigma: float):
    n = len(z_hat[t])
    tol = 0.031
    q_aux = [-(1 / sigma) * z_hat[t][i] for i in range(n)];
    Id = np.identity(n)
    Q_aux = [[1 / sigma * Id[i][j] for i in range(n)] for j in range(n)]
    A_aux = [[1] for i in range(n)]
    b = [1]
    Id = np.identity(n)
    G_aux = [[-Id[i][j] for i in range(n)] for j in range(n)]
    for i in range(n):
        G_aux[i] = G_aux[i] + [risk[i]]

    h_aux = [0] * n + [tol]

    Q = matrix(Q_aux, tc='d')
    q = matrix(q_aux, tc='d')
    A = matrix(A_aux, tc='d')
    b = matrix(b, tc='d')
    G = matrix(G_aux, tc='d')
    h = matrix(h_aux, tc='d')

    solvers.options['show_progress'] = False
    z_next = solvers.qp(Q, q, G, h, A, b)
    z_solution = [z_next['x'][i] for i in range(n)]

    return z_solution


def bilinear_min_obj(z_hat, risk, t: int, sigma: float):
    n = len(z_hat[t])
    q_aux = [-(1 / sigma) * z_hat[t][i] + risk[i] for i in range(n)]
    Id = np.identity(n)
    Q_aux = [[1 / sigma * Id[i][j] for i in range(n)] for j in range(n)]
    A_aux = [[1] for i in range(n)]
    b = [1]
    Id = np.identity(n)
    G_aux = [[-Id[i][j] for i in range(n)] for j in range(n)]
    h_aux = [0] * n

    Q = matrix(Q_aux, tc='d')
    q = matrix(q_aux, tc='d')
    A = matrix(A_aux, tc='d')
    b = matrix(b, tc='d')
    G = matrix(G_aux, tc='d')
    h = matrix(h_aux, tc='d')

    solvers.options['show_progress'] = False
    z_next = solvers.qp(Q, q, G, h, A, b);
    z_solution = [z_next['x'][i] for i in range(n)]
    return z_solution


def linear_min(f_next: List[float], g_next: List[float], risk: List[float]):
    n = len(f_next)
    A = [[1]*n]; b = [1]
    f = [f_next[i] + g_next[i] + risk[i] for i in range(n)]
    bound = [(0, None)] * (n)
    res = linprog(f, A_ub=A, b_ub=b, bounds=bound)
    return res.fun, res.x


def gap_primal_dual(W, Inf, T, risk, lambda_vec, portfolios: List[Portfolio], f_next, g_next, cte: float):
    w = 0  # max gap
    for i in range(len(portfolios) - 1):
        w_time, f, g = wasserstein(portfolios[i], portfolios[i + 1], cte)
        w = w + w_time * lambda_vec[i]

    W = W + [w]
    n = len(portfolios[0].flatten())
    primal_min = [0] * (T + 2)  # inf gap
    for t in range(1, T + 2):
        f_fix = [lambda_vec[t] * f_next[t][i] for i in range(n)]
        g_fix = [lambda_vec[t - 1] * g_next[t][i] for i in range(n)]
        primal_min[t], z_min = linear_min(f_fix, g_fix, risk)

    extremes = lambda_vec[0] * np.dot(f_next[0], portfolios[0].get_p_list()) + \
               lambda_vec[T + 1] * np.dot(portfolios[-1].get_p_list(), g_next[T + 2])
    Inf = Inf + [sum(primal_min) + extremes]
    return Inf, W, sum(primal_min) + extremes, w
