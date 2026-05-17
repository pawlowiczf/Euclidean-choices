from dataclasses import dataclass


@dataclass
class Candidate:
    id: int
    position: tuple[float, float]
    score: float = 0.0

    def reset_score(self):
        self.score = 0.0
