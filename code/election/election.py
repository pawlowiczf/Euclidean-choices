from candidate.candidate import Candidate
from voter.voter import Voter
from strategy.strategy import VotingStrategy


class Election:
    def __init__(self, candidates: list[Candidate], voters: list[Voter]):
        self.candidates = candidates
        self.voters = voters

    def run(self, strategy: VotingStrategy = None):
        self.reset_candidates_scores()
        id_to_candidate = {c.id: c for c in self.candidates}

        for voter in self.voters:
            points = voter.vote(self.candidates, strategy)
            for candidate_id, pts in points.items():
                id_to_candidate[candidate_id].score += pts

    def winner(self) -> Candidate:
        return max(self.candidates, key=lambda c: c.score)

    def compare_strategies(
        self, strategies: list[VotingStrategy]
    ) -> dict[VotingStrategy, Candidate]:
        results = {}
        for strategy in strategies:
            self.run(strategy)
            results[strategy] = self.winner()
        return results

    def reset_candidates_scores(self):
        for candidate in self.candidates:
            candidate.reset_score()
