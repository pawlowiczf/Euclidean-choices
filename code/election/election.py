from candidate.candidate import Candidate
from voter.voter import Voter
from strategy.strategy import VotingStrategy
from election.tally import Tally
from election.result import ElectionResult


class Election:
    def __init__(self, candidates: list[Candidate], voters: list[Voter]):
        self.candidates = candidates
        self.voters = voters

    def run(self, strategy: VotingStrategy) -> Tally:
        scores = {c.id: 0.0 for c in self.candidates}
        for voter in self.voters:
            points = voter.vote(self.candidates, strategy)
            for candidate_id, pts in points.items():
                scores[candidate_id] += pts
        return Tally(scores=scores)

    def compare_strategies(
        self, strategies: list[VotingStrategy], label: str | None = None
    ) -> ElectionResult:
        return ElectionResult(
            candidates=list(self.candidates),
            voters=list(self.voters),
            tallies={s: self.run(s) for s in strategies},
            label=label,
        )
