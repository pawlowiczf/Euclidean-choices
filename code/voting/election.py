"""Candidates, voters, and the rules run over them.

An election is a set of candidate positions, a set of voter positions, and a
:class:`~voting.profile.Profile` derived from them. Rules never see the
positions; the geometry's only job is to produce the rankings.

Candidates are identified by their **index** in the candidate list - in
``Tally.scores``, in ``winner_indices()``, in the LP's targets and in the plot
labels. ``Candidate.id`` is carried along for display, but nothing indexes by it.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from voting.profile import Profile
from voting.rules import CLASSIC_RULES, Rule


# eq=False keeps identity equality. The default would compare the ndarray
# positions, which returns an array rather than a bool and breaks any use of
# `==` or `list.index` on candidates.
@dataclass(eq=False)
class Candidate:
    id: int
    position: np.ndarray


@dataclass(eq=False)
class Voter:
    position: np.ndarray


@dataclass(frozen=True)
class Tally:
    """One rule's scores, indexed by the candidate's position in the list."""

    scores: np.ndarray
    rule_key: str = ""

    def __repr__(self) -> str:
        return f"Tally({self.rule_key!r}, winner={self.winner_index})"

    @property
    def winner_index(self) -> int:
        """The winner; ties fall back to the lower index, see :meth:`is_tie`."""
        return int(np.argmax(self.scores))

    @property
    def top_indices(self) -> list[int]:
        """Every candidate sharing the top score."""
        return [int(i) for i in np.flatnonzero(self.scores == self.scores.max())]

    @property
    def margin(self) -> float:
        """How far the winner is ahead of the runner-up; 0 exactly when tied."""
        if len(self.scores) < 2:
            return float("inf")
        best, second = np.sort(self.scores)[-1], np.sort(self.scores)[-2]
        return float(best - second)

    def is_tie(self) -> bool:
        """True if the top score is shared.

        A tied result has margin 0, so the LP cannot be asked to reproduce it as
        a strict win - the notebooks reject such draws before building a model.
        """
        return len(self.top_indices) > 1

    def winner(self, candidates: list) -> Candidate:
        return candidates[self.winner_index]


@dataclass(frozen=True)
class ElectionResult:
    """What several rules made of one election."""

    candidates: list
    voters: list
    tallies: dict
    label: str | None = None

    def __repr__(self) -> str:
        return (
            f"ElectionResult(label={self.label!r}, rules={sorted(self.tallies)}, "
            f"n_candidates={len(self.candidates)})"
        )

    @staticmethod
    def _key(rule) -> str:
        return rule if isinstance(rule, str) else rule.key

    def tally(self, rule) -> Tally:
        """The tally of one rule, by rule object or by key."""
        return self.tallies[self._key(rule)]

    def winner(self, rule) -> Candidate:
        return self.tally(rule).winner(self.candidates)

    def winner_index(self, rule) -> int:
        return self.tally(rule).winner_index

    def winners(self) -> dict:
        """``{rule key: winning Candidate}``."""
        return {key: t.winner(self.candidates) for key, t in self.tallies.items()}

    def winner_indices(self) -> dict:
        """``{rule key: winning candidate index}``.

        The same shape the LP takes as its target, so an observed outcome can be
        handed straight back to a model.
        """
        return {key: t.winner_index for key, t in self.tallies.items()}

    def has_tie(self) -> bool:
        """True if any rule's top score is shared."""
        return any(t.is_tie() for t in self.tallies.values())

    def rules_disagree(self) -> bool:
        """True if the rules did not all pick the same candidate."""
        return len(set(self.winner_indices().values())) > 1

    def winners_all_distinct(self) -> bool:
        """True if no two rules picked the same candidate.

        Stronger than :meth:`rules_disagree`, which only asks whether the rules
        failed to be unanimous. The two part company on the middling case: with
        three rules, two agreeing and one dissenting counts as disagreement but
        not as distinct.
        """
        indices = list(self.winner_indices().values())
        return len(set(indices)) == len(indices)

    def shared_winners(self) -> dict:
        """``{candidate index: the rule keys that picked it}``, repeats only.

        What :meth:`winners_all_distinct` returned False about.
        """
        by_candidate: dict = {}
        for key, index in self.winner_indices().items():
            by_candidate.setdefault(index, []).append(key)
        return {i: keys for i, keys in by_candidate.items() if len(keys) > 1}


class Election:
    """Candidates, voters, and the profile their positions induce."""

    def __init__(self, candidates: list, voters: list):
        self.candidates = _as_candidates(candidates)
        self.voters = _as_voters(voters)

    def __repr__(self) -> str:
        return f"Election(n_candidates={self.n_candidates}, n_voters={self.n_voters})"

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def n_voters(self) -> int:
        return len(self.voters)

    @cached_property
    def candidate_positions(self) -> np.ndarray:
        return np.array([c.position for c in self.candidates], dtype=float)

    @cached_property
    def voter_positions(self) -> np.ndarray:
        return np.array([v.position for v in self.voters], dtype=float)

    @cached_property
    def profile(self) -> Profile:
        """The rankings the positions induce - the only thing rules read."""
        return Profile.from_positions(self.voter_positions, self.candidate_positions)

    def run(self, rule: Rule) -> Tally:
        """Tally one rule."""
        return Tally(scores=np.asarray(rule.scores(self.profile)), rule_key=rule.key)

    def compare(self, rules=CLASSIC_RULES, label: str | None = None) -> ElectionResult:
        """Tally several rules over the same profile."""
        return ElectionResult(
            candidates=list(self.candidates),
            voters=list(self.voters),
            tallies={rule.key: self.run(rule) for rule in rules},
            label=label,
        )


def _as_candidates(candidates) -> list:
    """Accept Candidate objects, or bare positions to wrap in them."""
    candidates = list(candidates)
    if candidates and hasattr(candidates[0], "position"):
        return candidates
    return [
        Candidate(id=i, position=np.asarray(p, dtype=float))
        for i, p in enumerate(candidates)
    ]


def _as_voters(voters) -> list:
    """Accept Voter objects, or bare positions to wrap in them."""
    voters = list(voters)
    if voters and hasattr(voters[0], "position"):
        return voters
    return [Voter(position=np.asarray(p, dtype=float)) for p in voters]
