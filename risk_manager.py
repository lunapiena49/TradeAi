import cupy as cp

class QuantumRiskManager:
    def __init__(self):
        self.var_confidence = 0.99
        self.max_drawdown = -0.2

    def calculate_var(self, returns: cp.ndarray) -> float:
        sorted_returns = cp.sort(returns)
        cutoff = int((1 - self.var_confidence) * len(sorted_returns))
        return -sorted_returns[cutoff].get()
