from typing import List, Tuple, Optional

from src.entities import PortfolioData
"""
    The portfolio is represented by a class Portfolio composed by elements a list os lists of probabilities, thought as 
    probabilities in a grid. 
    portfolio.get_p(i,j) represents the percentage of investiments in an investiment class 
    {portfolios[t].get_esg(i,j).portfolios[t].get_sector(i,j)}, an element in R2 (sectors correspond to numbers)
    i,j in {0,...,N} represents the possible investiment classes.
    
    For each time t, portfolio.get_p_list() is a probability vector, where t in {0,...,T}.
    portfolio.get_p_list() is a vector such that 0 < portfolio.get_p_list()[i] < 1 and 
    sum(portfolio.get_p_list()) = 1 for each t. 
    
"""
class Portfolio:
    list: List[List[PortfolioData]] = [[]]

    @classmethod
    def empty_from_arrays(cls, sector_arr: List[int], esg_arr: List[float]) -> 'Portfolio':
        return Portfolio([[PortfolioData(esg, sector, 0, 0) for sector in sector_arr] for esg in esg_arr])

    @classmethod
    def empty_from_portfolio(cls, grid: 'Portfolio') -> 'Portfolio':
        return Portfolio([[PortfolioData(v.esg, v.sector, v.credit_spread, 0) for v in v_arr] for
                          v_arr in grid.get_matrix()])

    def __init__(self, arg: List[List[PortfolioData]]):
        n = len(arg)
        if n == 0:
            raise Exception('array needs at least one element')
        for i in range(n):
            if len(arg[i]) != n:
                raise Exception('array needs to be an N-by-N matrix')

        self.list = arg

    def get_len(self):
        return len(self.list)

    def get_matrix(self) -> List[List[PortfolioData]]:
        return self.list

    def flatten(self, reverse_axis: bool = False) -> List[PortfolioData]:
        n = self.get_len()
        return [self.list[i][j] if not reverse_axis else self.list[j][i] for i in range(n) for j in range(n)]

    def get(self, i: int, j: int) -> PortfolioData:
        return self.list[i][j]

    def get_esg(self, i: int, j: int) -> float:
        return self.get(i, j).esg

    def get_esg_list(self) -> List[float]:
        return [self.get_esg(i, 0) for i in range(self.get_len())]

    def get_unique_esg_list(self) -> List[float]:
        return list(set(self.get_esg_list()))

    def get_sector(self, i: int, j: int) -> float:
        return self.get(i, j).sector

    def get_sector_list(self) -> List[float]:
        return [self.get_sector(0, i) for i in range(self.get_len())]

    def get_unique_sector_list(self) -> List[float]:
        return list(set(self.get_sector_list()))

    def get_p_list(self) -> List[float]:
        return [v.p for v in self.flatten()]

    def get_p(self, i: int, j: int) -> Optional[float]:
        return self.get(i, j).p

    def set_p(self, i: int, j: int, value: Optional[float]) -> None:
        self.list[i][j].p = value

    def add_p(self, i: int, j: int, value: float) -> None:
        current_value = self.get(i, j).p
        if current_value is not None:
            self.set_p(i, j, current_value + value)
        else:
            self.set_p(i, j, value)

    def set_credit_spread(self, i: int, j: int, credit_spread: float) -> None:
        self.list[i][j].credit_spread = credit_spread
