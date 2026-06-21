import numpy as np

from strategy.strategy import VotingStrategy


class BordaCountStrategy(VotingStrategy):
    key = "borda"
    name = "Borda count"

    def tally_scores(self, distances: np.ndarray) -> np.ndarray:
        n_candidates = distances.shape[1]
        # rank 0 = nearest; nearest gets n-1 points, farthest gets 0.
        rank = np.argsort(np.argsort(distances, axis=1), axis=1)
        return ((n_candidates - 1) - rank).sum(axis=0).astype(float)
