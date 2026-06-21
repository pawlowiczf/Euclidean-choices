import numpy as np

from candidate.candidate import Candidate
from voter.voter import Voter
from strategy.strategy import VotingStrategy
from election.tally import Tally
from election.result import ElectionResult


class Election:
    def __init__(self, candidates: list[Candidate], voters: list[Voter]):
        self.candidates = candidates
        self.voters = voters

    def _distance_matrix(self) -> np.ndarray:
        """(n_voters, n_candidates) euclidean distances, candidate columns in
        self.candidates order. Computed once and shared across strategies."""
        voter_positions = np.array([v.position for v in self.voters])
        candidate_positions = np.array([c.position for c in self.candidates])
        return np.linalg.norm(
            voter_positions[:, None, :] - candidate_positions[None, :, :], axis=2
        )

    def _tally(self, strategy: VotingStrategy, distances: np.ndarray) -> Tally:
        scores = strategy.tally_scores(distances)
        return Tally(scores={c.id: float(s) for c, s in zip(self.candidates, scores)})

    def run(self, strategy: VotingStrategy) -> Tally:
        return self._tally(strategy, self._distance_matrix())

    def compare_strategies(
        self, strategies: list[VotingStrategy], label: str | None = None
    ) -> ElectionResult:
        distances = self._distance_matrix()
        return ElectionResult(
            candidates=list(self.candidates),
            voters=list(self.voters),
            tallies={s.key: self._tally(s, distances) for s in strategies},
            label=label,
        )
