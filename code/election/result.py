from dataclasses import dataclass

from candidate.candidate import Candidate
from voter.voter import Voter
from strategy.strategy import VotingStrategy
from election.tally import Tally


@dataclass(frozen=True)
class ElectionResult:
    candidates: list[Candidate]
    voters: list[Voter]
    tallies: dict[VotingStrategy, Tally]
    label: str | None = None

    def winner(self, strategy: VotingStrategy) -> Candidate:
        return self.tallies[strategy].winner(self.candidates)

    def winners(self) -> dict[VotingStrategy, Candidate]:
        return {s: t.winner(self.candidates) for s, t in self.tallies.items()}
