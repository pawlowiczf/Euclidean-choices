from dataclasses import dataclass

from candidate.candidate import Candidate


@dataclass(frozen=True)
class Tally:
    scores: dict[int, float]

    def winner(self, candidates: list[Candidate]) -> Candidate:
        winner_id = max(self.scores, key=self.scores.get)
        return next(c for c in candidates if c.id == winner_id)
