from abc import ABC, abstractmethod
import numpy as np


class VotingStrategy(ABC):
    @property
    @abstractmethod
    def key(self) -> str:
        """Short identifier of the strategy"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label for display"""
        pass

    def __str__(self) -> str:
        return self.name

    @abstractmethod
    def tally_scores(self, distances: np.ndarray) -> np.ndarray:
        """Vectorised tally over all voters at once.

        `distances` is a (n_voters, n_candidates) matrix of voter-candidate
        distances; return the total score per candidate as a length
        n_candidates array, with column order matching `distances`. Election
        computes the distance matrix once and calls this for each rule.
        """
        pass
