import numpy as np

from strategy.strategy import VotingStrategy


class VetoStrategy(VotingStrategy):
    key = "veto"
    name = "Veto rule"

    def tally_scores(self, distances: np.ndarray) -> np.ndarray:
        # Every voter vetoes (0 points) their farthest candidate, 1 to the rest;
        # a candidate's score is n_voters minus how often it was the farthest.
        n_voters, n_candidates = distances.shape
        vetoes = np.bincount(np.argmax(distances, axis=1), minlength=n_candidates)
        return (n_voters - vetoes).astype(float)
