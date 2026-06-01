import numpy as np

from strategy.strategy import VotingStrategy
from candidate.candidate import Candidate


class PluralityStrategy(VotingStrategy):
    @property
    def key(self) -> str:
        return "plurality"

    @property
    def name(self) -> str:
        return "Plurality rule"

    def choose(
        self, voter_position: np.ndarray, candidates: list[Candidate]
    ) -> dict[int, float]:
        distances = [
            np.linalg.norm(voter_position - np.array(c.position)) for c in candidates
        ]
        idx = int(np.argmin(distances))
        return {candidates[idx].id: 1.0}
