"""Preference profiles: who ranks the candidates in what order.

Every voting rule in this library reads a profile and nothing else. That is the
one structural idea worth stating: the Euclidean part of the project turns
positions into rankings, and from there the rules are ordinary social choice
functions that know nothing about geometry.

It matters for more than tidiness. A Condorcet method cannot even be *written*
against a matrix of voter-candidate distances without first rebuilding the
rankings from it - which is why the earlier version of this project could only
express plurality, Borda and veto. Once the profile is the input, Copeland,
Maxmin, Ranked pairs and STV are each a short, self-contained class.

A profile can be built four ways, one per source the project actually uses:

``from_positions``   Euclidean: each voter ranks candidates by distance.
``from_rankings``    An explicit list of rankings, one per voter.
``from_counts``      ``{(0, 1, 2): 5, ...}`` - the textbook notation, and the
                     shape the LP in :mod:`voting.lp` solves for.
``from_tournament``  McGarvey's construction: a profile realizing any desired
                     pattern of pairwise victories, cycles included.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

Ranking = tuple[int, ...]


@dataclass(frozen=True)
class Profile:
    """A complete set of voter rankings stored as a (n_voters, n_candidates) array.

    Each row contains candidate indices ordered from most to least preferred.
    For example, rankings[3] == [2, 0, 1] means that voter 3 prefers
    candidate 2, then candidate 0, then candidate 1.

    Candidates are represented by their indices throughout the library, including
    scores, winners, LP targets and plots.
    """

    rankings: np.ndarray
    n_candidates: int

    def __post_init__(self):
        if self.rankings.ndim != 2:
            raise ValueError(f"rankings must be 2-D, got shape {self.rankings.shape}")
        if self.rankings.shape[1] != self.n_candidates:
            raise ValueError(
                f"rankings are {self.rankings.shape[1]} wide but n_candidates "
                f"is {self.n_candidates}"
            )

    def __repr__(self) -> str:
        return f"Profile(n_voters={self.n_voters}, n_candidates={self.n_candidates})"

    @classmethod
    def from_positions(cls, voter_positions, candidate_positions) -> "Profile":
        """Rank candidates by Euclidean distance - the project's usual source."""
        voters = np.asarray(voter_positions, dtype=float)
        candidates = np.asarray(candidate_positions, dtype=float)
        distances = np.linalg.norm(voters[:, None, :] - candidates[None, :, :], axis=2)
        # np.argsort transforms geometric points into preferences (rankings).
        return cls(rankings=np.argsort(distances, axis=1), n_candidates=len(candidates))

    @classmethod
    def from_rankings(cls, rankings, n_candidates: int | None = None) -> "Profile":
        """One explicit ranking per voter."""
        array = np.asarray(rankings, dtype=int)
        if array.ndim == 1:
            array = array[None, :]
        return cls(
            rankings=array,
            n_candidates=n_candidates if n_candidates is not None else array.shape[1],
        )

    @classmethod
    def from_counts(cls, counts: dict, n_candidates: int | None = None) -> "Profile":
        """Build a profile from ``{ranking: voter count}``.

        Example::

            Profile.from_counts({(0, 1, 2): 5, (1, 2, 0): 4, (2, 0, 1): 3})
        """
        rows = []
        for ranking, count in counts.items():
            rows.extend([list(ranking)] * int(count))
        if not rows:
            width = n_candidates or 0
            return cls(rankings=np.empty((0, width), dtype=int), n_candidates=width)
        return cls.from_rankings(rows, n_candidates)

    @classmethod
    def from_tournament(cls, beats: np.ndarray) -> "Profile":
        """McGarvey's construction of a profile realizing ``beats``.

        ``beats[a, b]`` means that ``a`` should beat ``b`` head-to-head.
        For each edge ``a -> b``, two reverse rankings are added with ``a``
        immediately above ``b``. All other pairwise preferences cancel out.
        """
        beats = np.asarray(beats, dtype=bool)
        n_candidates = len(beats)
        rows = []
        for a in range(n_candidates):
            for b in range(n_candidates):
                if a == b or not beats[a, b]:
                    continue
                others = [c for c in range(n_candidates) if c not in (a, b)]
                rows.append([a, b] + others)
                rows.append(others[::-1] + [a, b])
        if not rows:
            return cls(
                rankings=np.empty((0, n_candidates), dtype=int),
                n_candidates=n_candidates,
            )
        return cls.from_rankings(rows, n_candidates)

    @property
    def n_voters(self) -> int:
        return len(self.rankings)

    def counts(self) -> dict:
        """``{ranking: how many voters hold it}`` - the inverse of from_counts."""
        tally: dict = {}
        for ranking in self.rankings:
            key = tuple(int(c) for c in ranking)
            tally[key] = tally.get(key, 0) + 1
        return tally

    def __add__(self, other: "Profile") -> "Profile":
        """Put two electorates together."""
        if other.n_candidates != self.n_candidates:
            raise ValueError("cannot add profiles over different candidate sets")
        return Profile(
            rankings=np.vstack([self.rankings, other.rankings]),
            n_candidates=self.n_candidates,
        )

    # What rules read:

    @cached_property
    def ranks(self) -> np.ndarray:
        """``(n_voters, M)`` where ``ranks[i, c]`` is candidate c's position.

        The inverse view of ``rankings``: ``rankings`` answers "who is 2nd?",
        ``ranks`` answers "where is candidate 2?". Scoring rules want the latter.
        """
        ranks = np.empty_like(self.rankings)
        positions = np.arange(self.n_candidates)
        for i, ranking in enumerate(self.rankings):
            # fancy indexing
            ranks[i, ranking] = positions
        return ranks

    @cached_property
    def pairwise(self) -> np.ndarray:
        """``(M, M)`` where ``pairwise[a, b]`` is how many voters rank a above b.

        The majority matrix. Every Condorcet method is a different way of reading
        it, and Borda happens to be its row sums.
        """
        ranks = self.ranks
        matrix = np.zeros((self.n_candidates, self.n_candidates), dtype=int)
        for a in range(self.n_candidates):
            for b in range(self.n_candidates):
                if a != b:
                    matrix[a, b] = np.count_nonzero(ranks[:, a] < ranks[:, b])
        return matrix

    @cached_property
    def margins(self) -> np.ndarray:
        """``(M, M)`` net majority: ``margins[a, b] > 0`` means a beats b."""
        return self.pairwise - self.pairwise.T

    def top_choices(self) -> np.ndarray:
        """``(n_voters,)`` each voter's most preferred candidate."""
        return self.rankings[:, 0]

    def beats(self) -> np.ndarray:
        """``(M, M)`` boolean majority graph, the tournament ``a beats b``."""
        return self.margins > 0

    # Condorcet:

    def condorcet_winner(self) -> int | None:
        """The candidate that beats every other head to head, if there is one.

        Often there is not - that is the whole interest of the subject.
        """
        beats = self.beats()
        for candidate in range(self.n_candidates):
            others = [c for c in range(self.n_candidates) if c != candidate]
            if all(beats[candidate, other] for other in others):
                return candidate
        return None

    def condorcet_loser(self) -> int | None:
        """The candidate that loses every head-to-head duel, if there is one."""
        beats = self.beats()
        for candidate in range(self.n_candidates):
            others = [c for c in range(self.n_candidates) if c != candidate]
            if all(beats[other, candidate] for other in others):
                return candidate
        return None

    def has_condorcet_cycle(self) -> bool:
        """True if the majority graph contains a cycle (a > b > c > a).

        Equivalent to there being no way to line the candidates up so that every
        majority points forwards.
        """
        beats = self.beats()
        # Transitive closure by repeated squaring of the reachability relation.
        reach = beats.copy()
        for _ in range(self.n_candidates):
            reach |= reach @ reach
        return bool(np.any(np.diagonal(reach)))
