import string
from random import random
from typing import List

import pandas as pandas

from src.entities import Portfolio


def load_portfolio_data(folder_path: string) -> (Portfolio, Portfolio):
    """
    Load the both the initial portfolio and reference portfolio from the memory

    :param folder_path: relative path of the files

    :return: a pair of portfolios, a initial and a final one
    """

    initial = _load_from_file(folder_path + "\\initial.csv")
    final = _load_from_file(folder_path + "\\final.csv")
    return initial, final


def _load_from_file(folder_path: string) -> Portfolio:

    df = pandas.read_csv(folder_path, sep=",")
    esg_list = list(df['ESG'].unique())
    esg_list.sort()
    sector_list = list(df['Sector'].unique())
    # sector_list.sort()

    portfolio = Portfolio.empty_from_arrays(sector_list, esg_list)

    for _, row in df.iterrows():
        i = esg_list.index(row['ESG'])
        j = sector_list.index(row['Sector'])
        credit_spread = row['Credit Spread']
        probability = row['Percentage']

        portfolio.set_credit_spread(i, j, credit_spread)
        portfolio.set_p(i, j, probability)

    return portfolio


def random_portfolios(base_portfolio: Portfolio, n: int, enable_random: bool = True) -> List[Portfolio]:
    """
    Generate portfolios for each time stamp

    :param base_portfolio: Base portfolio to copy esg and sectors from
    :param n: Number of random portfolios to copy
    :param enable_random: Enable random number generator for the distributions or use a cte value

    :return: n-copies of the original portfolio with randomized probabilities
    """

    return [random_portfolio(base_portfolio, enable_random) for _ in range(n)]


def random_portfolio(base_portfolio: Portfolio, enable_random: bool = True) -> Portfolio:
    """
    Generate portfolios for each time stamp

    :param base_portfolio: Base portfolio to copy esg and sectors from
    :param enable_random: Enable random number generator for the distributions or use a cte value

    :return: Copy of the portfolio with randomized probabilities
    """
    portfolio = Portfolio.empty_from_portfolio(base_portfolio)

    n = portfolio.get_len()
    prob = [random() if enable_random else 1 for _ in range(n ** 2)]
    prob_sum = sum(prob)

    for k, point in enumerate(portfolio.flatten()):
        point.p = prob[k]/prob_sum

    return portfolio
