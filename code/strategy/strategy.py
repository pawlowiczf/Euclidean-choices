from abc import ABC, abstractmethod
import numpy as np

from candidate.candidate import Candidate


class VotingStrategy(ABC):
    @abstractmethod
    def choose(
        self, voter_position: np.ndarray, candidates: list[Candidate]
    ) -> dict[int, float]:
        pass
