from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    id: int
    position: tuple[float, float]
