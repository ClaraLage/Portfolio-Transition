from typing import Optional


class PortfolioData:
    esg: float = 0
    sector: int = 0
    credit_spread: float = 0
    p: Optional[float] = 0

    def __init__(self, esg: float, sector: int, credit_spread: float, p: Optional[float]):
        self.esg = esg
        self.sector = sector
        self.credit_spread = credit_spread
        self.p = p

    def get_esg(self) -> float:
        return self.esg

    def get_sector(self) -> int:
        return self.sector

    def get_credit_spread(self) -> float:
        return self.credit_spread

    def get_p(self) -> Optional[float]:
        return self.p
