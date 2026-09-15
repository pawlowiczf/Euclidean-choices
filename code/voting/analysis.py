"""Measurements over elections, and the classical criteria.

Two kinds of thing live here. The distance measurements are geometric - they are
the project's central quantity, "how far apart do different rules put their
winners". The criteria are ordinal, and answer the questions the theory notes
ask: does this rule pick the Condorcet winner here, and is this one of those
profiles where plurality crowns the candidate everybody else beats.
"""

from itertools import combinations

import numpy as np

from voting.profile import Profile
from voting.rules import PluralityRule, Rule


def mean_pairwise_distance(points) -> float:
    """Average distance over every pair of points; 0 for fewer than two."""
    points = [np.asarray(p, dtype=float) for p in points]
    if len(points) < 2:
        return 0.0
    return float(np.mean([np.linalg.norm(a - b) for a, b in combinations(points, 2)]))


def find_farthest_pair(points) -> tuple[int, int]:
    """Indices of the two points furthest apart."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError(f"need at least 2 points, got {len(points)}")
    return max(
        combinations(range(len(points)), 2),
        key=lambda pair: np.linalg.norm(points[pair[0]] - points[pair[1]]),
    )


def find_farthest_triple(points) -> tuple[int, int, int]:
    """Indices of the three points with the largest perimeter."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError(f"need at least 3 points, got {len(points)}")

    def perimeter(triple):
        return sum(
            np.linalg.norm(points[a] - points[b]) for a, b in combinations(triple, 2)
        )

    return max(combinations(range(len(points)), 3), key=perimeter)


class ResultsAnalyzer:
    """Summaries across a list of :class:`~voting.election.ElectionResult`."""

    def __init__(self, results: list):
        self.results = results

    def __repr__(self) -> str:
        return f"ResultsAnalyzer(n_results={len(self.results)})"

    def mean_pairwise_winner_distance(self, result) -> float:
        """How far apart this election's various winners stand."""
        winners = list(result.winners().values())
        return mean_pairwise_distance([w.position for w in winners])

    def winner_distance_series(self) -> list[float]:
        """That distance for every result, ready to histogram."""
        return [self.mean_pairwise_winner_distance(r) for r in self.results]

    def disagreement_rate(self) -> float:
        """Fraction of elections where the rules did not all agree."""
        if not self.results:
            return 0.0
        return sum(r.rules_disagree() for r in self.results) / len(self.results)


def satisfies_condorcet(rule: Rule, profile: Profile) -> bool | None:
    """Did ``rule`` elect the Condorcet winner?

    ``None`` when the profile has no Condorcet winner, which is the interesting
    case and not a failure of the rule.
    """
    expected = profile.condorcet_winner()
    if expected is None:
        return None
    return rule.winner(profile) == expected


def satisfies_majority(rule: Rule, profile: Profile) -> bool | None:
    """Did ``rule`` elect the candidate ranked first by an outright majority?

    ``None`` when nobody holds such a majority. Plurality always passes this;
    Borda need not.
    """
    tops = profile.top_choices()
    counts = np.bincount(tops, minlength=profile.n_candidates)
    leader = int(np.argmax(counts))
    if counts[leader] * 2 <= profile.n_voters:
        return None
    return rule.winner(profile) == leader


def is_borda_paradox(profile: Profile, rule: Rule | None = None) -> bool:
    """Does the rule's winner lose every head-to-head duel?

    The paradox the notes name: a candidate wins on first preferences while
    being the one every other candidate beats one on one. Defaults to plurality,
    which is where it classically shows up.
    """
    loser = profile.condorcet_loser()
    if loser is None:
        return False
    rule = rule if rule is not None else PluralityRule()
    return rule.winner(profile) == loser
