import sys

from src.data import load_portfolio_data
from src.entities import PrimalDualParameters
from src.plot import plot_result
from src.primal_dual import run_primal_dual_test

default_parameters: PrimalDualParameters = PrimalDualParameters(T=1, cte=4, tau=200, min_gap=0.1,
                                                                max_iterations=500)

default_flags: dict = {
    'print': True,
    'random': False,
    'verbose': True
}


def test_simple_portfolio():
    flags = _get_flags()
    z_initial, z_final = load_portfolio_data(".\\data\\simple_portfolio")
    portfolios, _, _, _, risk = run_primal_dual_test(default_parameters, z_initial, z_final, enable_random=flags['random'], verbose=flags['verbose'])

    if flags['print']:
        plot_result(portfolios, risk)


def test_portfolio():
    flags = _get_flags()
    z_initial, z_final = load_portfolio_data(".\\data\\normal_portfolio")
    portfolios, _, _, _, risk = run_primal_dual_test(default_parameters, z_initial, z_final, enable_random=flags['random'], verbose=flags['verbose'])

    if flags['print']:
        plot_result(portfolios, risk)


def _get_flags() -> dict:
    args = sys.argv
    flags_copy = default_flags.copy()
    for key in flags_copy.keys():
        if key in args:
            flags_copy[key] = args[key]

    return flags_copy
