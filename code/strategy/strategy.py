from abc import ABC, abstractmethod
import numpy as np

from candidate.candidate import Candidate


class VotingStrategy(ABC):
    @property
    @abstractmethod
    def key(self) -> str:
        """Short, stable identifier (e.g. "plurality"). Used as a dict key and as
        the rule selector understood by the LP models."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label for display (e.g. "Plurality rule")."""
        pass

    def __str__(self) -> str:
        return self.name

    @abstractmethod
    def choose(
        self, voter_position: np.ndarray, candidates: list[Candidate]
    ) -> dict[int, float]:
        pass
