from typing import List


class PrimalDualParameters:
    T: int  # time steps
    cte: float  # normalization constant of transport cost
    tau: float  # parameter of primal dual algorithm
    max_iterations: int  # max iterations
    min_gap: float
    weight_vector: List[float] = []

    def __init__(self, T: int, cte: float, tau: float, max_iterations: int, min_gap: float,
                 weight_vector=None):
        self.T = T
        self.cte = cte
        self.tau = tau
        self.min_gap = min_gap
        self.max_iterations = max_iterations
        self.weight_vector = weight_vector if weight_vector is not None else [1 / (T + 2)] * (T + 4)

    def as_tuple(self):
        return self.T, self.cte, self.tau, self.min_gap, self.max_iterations, self.weight_vector
