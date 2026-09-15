"""Voting rules.

Every rule answers the same question - ``scores(profile)``, higher is better -
so the winner is always an argmax and nothing outside this module needs to know
which rule it is holding.

Rules come in three families, and the family is chosen by *what the rule reads*:

``ScoringRule``     a weight per rank. Plurality is ``(1, 0, ..., 0)``, veto is
                    ``(1, ..., 1, 0)``, Borda is ``(M-1, ..., 1, 0)``. Writing
                    them as one family is not just tidiness: the LP in
                    :mod:`voting.lp` builds its constraints from the very same
                    ``weights()``, so "what Borda means" is written down once.
``CondorcetRule``   the pairwise majority matrix. Copeland, Maxmin and Ranked
                    pairs differ only in how they read it.
``InstantRunoff``   the profile round by round, eliminating as it goes.

Adding a rule
-------------
Subclass the family that fits and give it a ``key``, a ``name`` and one method::

    class NansonRule(CondorcetRule):
        key, name = "nanson", "Nanson's method"

        def scores_from_pairwise(self, pairwise):
            ...  # (M,) array, higher is better

    register_rule(NansonRule())

That is the whole contract. The new rule then works in ``Election.compare``, in
the result objects, in the plot legends and in lookup by key. A scoring rule
additionally becomes usable as an LP target for free, because the LP reads
``weights()``.
"""

from abc import ABC, abstractmethod

import numpy as np

from voting.profile import Profile


class Rule(ABC):
    """A voting rule: a profile in, one score per candidate out."""

    key: str = ""
    name: str = ""

    @abstractmethod
    def scores(self, profile: Profile) -> np.ndarray:
        """``(n_candidates,)`` score per candidate; higher is better."""

    def winner(self, profile: Profile) -> int:
        """Index of the winning candidate; ties break towards the lower index."""
        return int(np.argmax(self.scores(profile)))

    def winners(self, profile: Profile) -> list[int]:
        """Every candidate tied at the top - the set-valued voting system."""
        scores = self.scores(profile)
        return [int(i) for i in np.flatnonzero(scores == scores.max())]

    def social_ranking(self, profile: Profile) -> list[int]:
        """All candidates ordered best to worst - the social welfare function."""
        return [int(i) for i in np.argsort(-self.scores(profile), kind="stable")]

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ScoringRule(Rule):
    """Awards ``weights(M)[r]`` points to the candidate a voter ranks ``r``-th."""

    @abstractmethod
    def weights(self, n_candidates: int) -> np.ndarray:
        """Points for rank 0 (most preferred) through rank M-1."""

    def scores(self, profile: Profile) -> np.ndarray:
        weights = np.asarray(self.weights(profile.n_candidates))
        # ranks[i, c] is candidate c's position for voter i, so indexing the
        # weights by it gives the points that voter awards each candidate.
        return weights[profile.ranks].sum(axis=0)


class PluralityRule(ScoringRule):
    """One point to each voter's favourite candidate."""

    key, name = "plurality", "Plurality rule"

    def weights(self, n_candidates: int) -> np.ndarray:
        weights = np.zeros(n_candidates, dtype=int)
        weights[0] = 1
        return weights


class VetoRule(ScoringRule):
    """One point to everyone except each voter's least preferred candidate."""

    key, name = "veto", "Veto rule"

    def weights(self, n_candidates: int) -> np.ndarray:
        weights = np.ones(n_candidates, dtype=int)
        weights[-1] = 0
        return weights


class BordaRule(ScoringRule):
    """``M-1`` points for first place, down to 0 for last."""

    key, name = "borda", "Borda count"

    def weights(self, n_candidates: int) -> np.ndarray:
        return np.arange(n_candidates - 1, -1, -1)


class KApprovalRule(ScoringRule):
    """One point to each voter's ``k`` most preferred candidates."""

    def __init__(self, k: int):
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k
        self.key = f"approval{k}"
        self.name = f"{k}-approval"

    def weights(self, n_candidates: int) -> np.ndarray:
        weights = np.zeros(n_candidates, dtype=int)
        weights[: min(self.k, n_candidates)] = 1
        return weights

    def __repr__(self) -> str:
        return f"KApprovalRule(k={self.k})"


class DowdallRule(ScoringRule):
    """``1/(r+1)`` points for rank ``r`` - Borda's Nauruan cousin."""

    key, name = "dowdall", "Dowdall"

    def weights(self, n_candidates: int) -> np.ndarray:
        return 1.0 / np.arange(1, n_candidates + 1)


class CustomScoringRule(ScoringRule):
    """A scoring rule from a weight vector you supply.

    ``weights`` is either a fixed sequence or a function of the candidate
    count::

        borda_like = CustomScoringRule(lambda m: np.arange(m)[::-1], "mine", "Mine")
        top_two = CustomScoringRule([3, 1, 0], "top2", "Top two")
    """

    def __init__(self, weights, key: str, name: str | None = None):
        self._weights = weights
        self.key = key
        self.name = name or key

    def weights(self, n_candidates: int) -> np.ndarray:
        values = (
            self._weights(n_candidates) if callable(self._weights) else self._weights
        )
        values = np.asarray(values)
        if len(values) != n_candidates:
            raise ValueError(
                f"{self.key!r} supplies {len(values)} weights but the election "
                f"has {n_candidates} candidates"
            )
        return values

    def __repr__(self) -> str:
        return f"CustomScoringRule(key={self.key!r})"


class CondorcetRule(Rule):
    """A rule defined on the pairwise majority matrix."""

    @abstractmethod
    def scores_from_pairwise(self, pairwise: np.ndarray) -> np.ndarray:
        """``(M,)`` score per candidate, read off the ``(M, M)`` matrix."""

    def scores(self, profile: Profile) -> np.ndarray:
        return self.scores_from_pairwise(profile.pairwise)


class CopelandRule(CondorcetRule):
    """Head-to-head wins minus head-to-head losses."""

    key, name = "copeland", "Copeland"

    def scores_from_pairwise(self, pairwise: np.ndarray) -> np.ndarray:
        wins = (pairwise > pairwise.T).sum(axis=1)
        losses = (pairwise < pairwise.T).sum(axis=1)
        return wins - losses


class MaxminRule(CondorcetRule):
    """Worst head-to-head support, maximised."""

    key, name = "maxmin", "Maxmin"

    def scores_from_pairwise(self, pairwise: np.ndarray) -> np.ndarray:
        support = pairwise.astype(float).copy()
        np.fill_diagonal(support, np.inf)  # a candidate does not face itself
        return support.min(axis=1)


class RankedPairsRule(CondorcetRule):
    """Lock in majorities strongest first, skipping any that closes a cycle.

    The score is how many candidates a candidate ends up above in the locked
    order, so the argmax is the Ranked Pairs winner and the whole social ranking
    is readable off the result. Equal margins are processed in index order.
    """

    key, name = "ranked_pairs", "Ranked pairs"

    def scores_from_pairwise(self, pairwise: np.ndarray) -> np.ndarray:
        n_candidates = len(pairwise)
        margins = pairwise - pairwise.T

        majorities = [
            (a, b)
            for a in range(n_candidates)
            for b in range(n_candidates)
            if a != b and margins[a, b] > 0
        ]
        majorities.sort(key=lambda pair: (-margins[pair], pair))

        # above[x, y] means x has been locked above y, directly or transitively.
        above = np.eye(n_candidates, dtype=bool)
        for a, b in majorities:
            if above[b, a]:
                continue  # b already stands above a, so this would close a cycle
            higher = np.flatnonzero(above[:, a])
            lower = np.flatnonzero(above[b, :])
            above[np.ix_(higher, lower)] = True

        return above.sum(axis=1) - 1


class InstantRunoffRule(Rule):
    """Single-winner STV: repeatedly drop the candidate with fewest first places.

    The score is the round a candidate survived to, so the last one standing is
    the argmax and the full elimination order is readable off the scores. Ties
    for fewest first places are broken towards the lower index.
    """

    key, name = "irv", "Instant-runoff (STV)"

    def scores(self, profile: Profile) -> np.ndarray:
        n_candidates = profile.n_candidates
        alive = np.ones(n_candidates, dtype=bool)
        survived = np.zeros(n_candidates, dtype=int)

        for round_number in range(n_candidates - 1):
            tally = np.zeros(n_candidates, dtype=float)
            for ranking in profile.rankings:
                for candidate in ranking:
                    if alive[candidate]:
                        tally[candidate] += 1
                        break
            tally[~alive] = np.inf
            loser = int(np.argmin(tally))
            alive[loser] = False
            survived[loser] = round_number

        survived[alive] = n_candidates - 1
        return survived


#: The rules the project started with, in their usual order.
CLASSIC_RULES = (PluralityRule(), BordaRule(), VetoRule())

RULES: dict = {
    rule.key: rule
    for rule in (
        PluralityRule(),
        BordaRule(),
        VetoRule(),
        DowdallRule(),
        CopelandRule(),
        MaxminRule(),
        RankedPairsRule(),
        InstantRunoffRule(),
    )
}


def register_rule(rule: Rule, replace: bool = False) -> Rule:
    """Make ``rule`` findable by key.

    Only needed for lookup *by key* - plot legends, and anything that stores a
    result and reads it back. Passing a rule object straight to an election
    always works without registering it.
    """
    if not rule.key:
        raise ValueError(f"{type(rule).__name__} needs a key before it can register")
    if rule.key in RULES and not replace:
        raise ValueError(
            f"key {rule.key!r} already belongs to {type(RULES[rule.key]).__name__}; "
            "pass replace=True to override"
        )
    RULES[rule.key] = rule
    return rule


def rule_for_key(key: str) -> Rule:
    """Look a rule up by key, understanding ``approvalK`` as k-approval."""
    if key in RULES:
        return RULES[key]
    if key.startswith("approval") and key[len("approval") :].isdigit():
        return KApprovalRule(int(key[len("approval") :]))
    raise KeyError(f"unknown rule {key!r}; known rules: {sorted(RULES)}")


def rule_name(key: str) -> str:
    """Readable name for a key, falling back to the key itself."""
    try:
        return rule_for_key(key).name
    except KeyError:
        return key
