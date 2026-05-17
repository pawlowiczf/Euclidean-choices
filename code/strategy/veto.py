import numpy as np

from strategy.strategy import VotingStrategy
from candidate.candidate import Candidate


class VetoStrategy(VotingStrategy):
    def __str__(self):
        return "Veto rule"

    def choose(
        self, voter_position: np.ndarray, candidates: list[Candidate]
    ) -> dict[int, float]:
        distances = [
            np.linalg.norm(voter_position - np.array(c.position)) for c in candidates
        ]
        veto_idx = int(np.argmax(distances))
        return {c.id: (0.0 if i == veto_idx else 1.0) for i, c in enumerate(candidates)}
