from dataclasses import dataclass

from candidate.candidate import Candidate


@dataclass(frozen=True)
class Tally:
    scores: dict[int, float]

    def winner(self, candidates: list[Candidate]) -> Candidate:
        winner_id = max(self.scores, key=self.scores.get)
        return next(c for c in candidates if c.id == winner_id)

    def is_tie(self) -> bool:
        """True if the top score is shared by more than one candidate.

        A tie means winner() falls back to id order; such a result has margin 0
        and cannot be enforced as a strict winner by the LP model.
        """
        best = max(self.scores.values())
        return sum(1 for s in self.scores.values() if s == best) > 1
