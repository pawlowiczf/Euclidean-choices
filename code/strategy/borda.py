import numpy as np

from strategy.strategy import VotingStrategy
from candidate.candidate import Candidate


class BordaCountStrategy(VotingStrategy):
    def __str__(self):
        return "Borda count"

    def choose(
        self, voter_position: np.ndarray, candidates: list[Candidate]
    ) -> dict[int, float]:
        distances = [
            np.linalg.norm(voter_position - np.array(c.position)) for c in candidates
        ]

        ranking = np.argsort(distances)
        n = len(candidates)

        return {
            candidates[int(candidate_idx)].id: (n - rank - 1)
            for rank, candidate_idx in enumerate(ranking)
        }
