import string
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src.entities import Portfolio



def plot_portfolio(portfolio: Portfolio, color: string,name):
    esg_list = [v.esg for v in portfolio.flatten()]
    sector_list = [v.sector for v in portfolio.flatten()]
    plt.plot(esg_list, sector_list, 'bo', markersize=0.5)
    plt.title(name)
    for v in portfolio.flatten():
        
        plt.plot(v.esg, v.sector, 'o', color=color, markersize=300 * v.p)

    return

def expected_esg(portfolio: Portfolio):
    expected_esg = sum([portfolio.get_esg(i,j)*portfolio.get_p(i,j) for i in range(portfolio.get_len()) for j in range(portfolio.get_len())])
    return expected_esg

def plot_result(portfolios: List[Portfolio], risk: List[float]):
    sectors = ['Transportation', 'Electronic Technology', 'Health Technology', 'Utilities', 'Non-Energy Minerals',
               'Producer Manufacturing', 'Health Services', 'Energy Minerals', 'Consumer Durables', 'Communications']

    all_sectors = portfolios[0].get_sector_list()

    for k in range(len(portfolios)):
        name = 'Initial Portfolio' if k == 0 else 'Final Portfolio' if k == len(
            portfolios) - 1 else 'Intermediate Portfolio {0}'.format(k)

       # plt.setp(title,title=name)
        fig = plt.figure(k)
        plt.yticks(ticks=all_sectors, labels=[sectors[i] for i in all_sectors])
        plot_portfolio(portfolios[k], 'mediumseagreen',name)

    plt.show()

    # Plot risk trajectory
    fig5 = plt.figure(5)
    plt.plot(range(len(portfolios)), [np.dot(risk, portfolio.get_p_list()) for portfolio in portfolios], label="MMK Solution Path")
    plt.plot([0, len(portfolios)-1], [np.dot(risk, portfolios[0].get_p_list()),
                                      np.dot(risk, portfolios[-1].get_p_list())], '--', label="Linear Path")
    plt.legend()
    plt.xticks([i for i in range(len(portfolios))], ['{0}'.format(k-1) for k in range(len(portfolios))])
    plt.ylabel('VaR', fontsize=15)
    plt.xlabel('Time', fontsize=15)

    plt.show()
    
    #Plot ESG trajectory
    fig6 = plt.figure(6)
    plt.plot(range(len(portfolios)), [expected_esg(portfolio) for portfolio in portfolios], label="MMK Solution Path")
    plt.plot([0, len(portfolios)-1], [expected_esg(portfolios[0]),
                                      expected_esg(portfolios[-1])], '--', label="Linear Path")
    plt.legend()
    plt.xticks([i for i in range(len(portfolios))], ['{0}'.format(k-1) for k in range(len(portfolios))])
    plt.ylabel('ESG', fontsize=15)
    plt.xlabel('Time', fontsize=15)

    plt.show()
