# The Trader agent state

import numpy as np
from mesa import Agent


class Trader(Agent):
    def __init__(self, model, node):
        super().__init__(model)
        self.node = node

    @property
    def strategy(self):
        return "chartist" if self.model.is_chart[self.node] else "fundamentalist"

    @property
    def equity(self):
        return float(self.model.equity[self.node])

    @property
    def leverage(self):
        e = self.model.equity[self.node]
        return float(self.model.shares[self.node] * self.model.price / e) if e > 0 else np.inf
