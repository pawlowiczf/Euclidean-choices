import numpy as np

from strategy.strategy import VotingStrategy


class PluralityStrategy(VotingStrategy):
    key = "plurality"
    name = "Plurality rule"

    def tally_scores(self, distances: np.ndarray) -> np.ndarray:
        # One vote each to the nearest candidate; count votes per candidate.
        nearest = np.argmin(distances, axis=1)
        return np.bincount(nearest, minlength=distances.shape[1]).astype(float)
