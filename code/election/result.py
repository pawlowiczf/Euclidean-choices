from dataclasses import dataclass

from candidate.candidate import Candidate
from voter.voter import Voter
from strategy.strategy import VotingStrategy
from election.tally import Tally


@dataclass(frozen=True)
class ElectionResult:
    candidates: list[Candidate]
    voters: list[Voter]
    tallies: dict[str, Tally]
    label: str | None = None

    def winner(self, strategy: VotingStrategy) -> Candidate:
        return self.tallies[strategy.key].winner(self.candidates)

    def winners(self) -> dict[str, Candidate]:
        return {key: t.winner(self.candidates) for key, t in self.tallies.items()}
