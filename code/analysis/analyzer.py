import numpy as np
from itertools import combinations

from election.result import ElectionResult


class ResultsAnalyzer:
    def __init__(self, results: list[ElectionResult]):
        self.results = results

    def mean_pairwise_winner_distance(self, result: ElectionResult) -> float:
        winners = list(result.winners().values())
        if len(winners) < 2:
            return 0.0
        positions = [np.array(w.position) for w in winners]
        distances = [np.linalg.norm(a - b) for a, b in combinations(positions, 2)]
        return float(np.mean(distances))

    def winner_distance_series(self) -> list[float]:
        return [self.mean_pairwise_winner_distance(r) for r in self.results]
