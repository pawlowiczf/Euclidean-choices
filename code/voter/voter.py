from abc import ABC, abstractmethod
import numpy as np

from candidate.candidate import Candidate
from strategy.strategy import VotingStrategy


class Voter:
    def __init__(self, position: np.ndarray, strategy: VotingStrategy = None):
        self.position = position
        self.strategy = strategy

    def vote(self, candidates: list[Candidate], strategy: VotingStrategy = None) -> dict[int, float]:
        return (strategy or self.strategy).choose(self.position, candidates)
