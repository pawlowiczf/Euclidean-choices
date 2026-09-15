"""Integer programs that build an electorate electing chosen winners.

The unknown is one count per ranking: ``x[sigma]`` voters hold ranking
``sigma``. A scoring rule's tally is then a linear function of ``x``, so
"candidate ``w`` wins under rule ``R``" is simply

    score_R(w) >= score_R(j) + 1    for every other candidate j

and the coefficients come from ``R.weights()``. That is why this module contains
no per-rule code at all: plurality, Borda, veto, k-approval and any scoring rule
added later are targets for free. Being a Condorcet winner is linear in ``x``
too, and is available as a separate target.

Which rankings are allowed
--------------------------
Not every one of the ``M!`` permutations can be produced by a point in the
plane. The realizable ones are found by scattering points over ``bounds`` and
keeping the rankings that turn up (:func:`sample_regions`). This is simple and
works for any number of candidates, but it is a sample: a region too thin to be
hit is treated as unrealizable, and the model may then report ``Infeasible`` for
a target that is in fact reachable. Raising ``pool_size`` makes that rarer.
"""

from typing import Callable

import numpy as np
from pulp import PULP_CBC_CMD, LpMinimize, LpProblem, LpStatus, LpVariable, lpSum

from voting.profile import Profile, Ranking
from voting.rules import ScoringRule, rule_for_key

#: How the model chooses *which* feasible electorate to return.
#:
#: ``feasibility`` any solution at all
#: ``minmax``      make the fullest region as small as possible
#: ``maxmin``      make the emptiest region as large as possible (spread out)
#: ``range``       squeeze the gap between fullest and emptiest
#: ``min_total``   use as few voters as possible
OBJECTIVES = ("feasibility", "minmax", "maxmin", "range", "min_total")


def sample_regions(
    candidate_positions,
    bounds: tuple[float, float] = (-10.0, 10.0),
    pool_size: int = 300_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """``{ranking: the sampled points that induce it}``.

    Scatters ``pool_size`` points over the square ``bounds`` and groups them by
    the ranking they produce. The keys are the rankings the LP may use; the
    values let :meth:`LpModel.generate_voter_positions` turn a solution back into
    voters standing in the right places.
    """
    rng = rng if rng is not None else np.random.default_rng()
    candidates = np.asarray(candidate_positions, dtype=float)

    points = rng.uniform(bounds[0], bounds[1], size=(pool_size, 2))
    rankings = Profile.from_positions(points, candidates).rankings

    regions: dict = {}
    for point, ranking in zip(points, rankings):
        regions.setdefault(tuple(int(c) for c in ranking), []).append(point)
    return {ranking: np.array(pts) for ranking, pts in regions.items()}


class LpModel:
    """Find an electorate under which the given rules elect the given winners.

    Two modes, picked by which argument you pass:

    *Build one from nothing* - ``n_voters`` fixes the size of the electorate::

        LpModel(candidates, winners={"plurality": 0, "borda": 3}, n_voters=1000)

    *Extend an existing one* - ``base_voters`` are fixed and counted, and at most
    ``max_added`` further voters may be placed::

        LpModel(candidates, winners={...}, base_voters=voters, max_added=500)

    ``winners`` maps a rule key to the candidate index that must win it. Pass
    ``condorcet_winner=j`` to additionally require that ``j`` beats every other
    candidate head to head.
    """

    def __init__(
        self,
        candidates: list,
        winners: dict | None = None,
        *,
        n_voters: int | None = None,
        base_voters: list | None = None,
        max_added: int | None = None,
        condorcet_winner: int | None = None,
        bounds: tuple[float, float] = (-10.0, 10.0),
        rng: np.random.Generator | None = None,
        objective: str = "feasibility",
        pool_size: int = 300_000,
    ):
        if objective not in OBJECTIVES:
            raise ValueError(
                f"objective must be one of {OBJECTIVES}, got {objective!r}"
            )
        if (n_voters is None) == (max_added is None):
            raise ValueError(
                "pass exactly one of n_voters (build an electorate) or "
                "max_added (extend the one in base_voters)"
            )

        self.candidates = candidates
        self.candidate_positions = np.array(
            [getattr(c, "position", c) for c in candidates], dtype=float
        )
        self.n_candidates = len(self.candidate_positions)
        self.winners = dict(winners or {})
        self.condorcet_winner = condorcet_winner
        self.n_voters = n_voters
        self.max_added = max_added
        self.bounds = bounds
        self.objective = objective
        self.pool_size = pool_size
        self.rng = rng if rng is not None else np.random.default_rng()

        self._check_targets()

        self.voters = base_voters
        self.base_profile = None
        if base_voters is not None:
            positions = np.array(
                [getattr(v, "position", v) for v in base_voters], dtype=float
            )
            self.base_profile = Profile.from_positions(
                positions, self.candidate_positions
            )

        self.model: LpProblem | None = None
        self.status: str | None = None
        self.variables: dict | None = None
        self._regions: dict | None = None

        # Extra constraints applied on top of the winner constraints in build().
        # Each callable receives the model and adds constraints through
        # `model.model += ...`, using `model.variables` (one per ranking).
        self.extra_constraints: list[Callable[["LpModel"], None]] = []

    def __repr__(self) -> str:
        return (
            f"LpModel(n_candidates={self.n_candidates}, targets={self.winners}, "
            f"objective={self.objective!r}, status={self.status!r})"
        )

    def _check_targets(self) -> None:
        """Reject bad targets here rather than deep inside build().

        An out-of-range index would otherwise raise an obscure ``IndexError``,
        and a negative one is worse: it quietly asks a candidate to beat itself,
        and the model comes back ``Infeasible`` - which reads like a real answer.
        """
        targets = dict(self.winners)
        if self.condorcet_winner is not None:
            targets["condorcet"] = self.condorcet_winner

        for key, index in targets.items():
            if key != "condorcet":
                rule = rule_for_key(key)  # raises KeyError listing known rules
                if not isinstance(rule, ScoringRule):
                    raise TypeError(
                        f"{rule.name} is not a scoring rule, so its winner is not a "
                        "linear function of the ranking counts and the LP cannot "
                        "target it; scoring rules and condorcet_winner can be"
                    )
            if not isinstance(index, (int, np.integer)):
                raise TypeError(f"target for {key!r} must be a candidate index")
            if not 0 <= index < self.n_candidates:
                raise ValueError(
                    f"target for {key!r} is candidate {index}, outside the valid "
                    f"range 0..{self.n_candidates - 1}"
                )

    # Regions:

    @property
    def regions(self) -> dict:
        """``{ranking: sampled points}`` for the realizable rankings."""
        if self._regions is None:
            self._regions = sample_regions(
                self.candidate_positions, self.bounds, self.pool_size, self.rng
            )
        return self._regions

    @property
    def rankings(self) -> list[Ranking]:
        """The rankings the LP is allowed to place voters into."""
        return sorted(self.regions)

    # Building:

    def add_constraint(self, constraint: Callable[["LpModel"], None]) -> None:
        """Register an extra constraint, applied on the next :meth:`build`.

        Called as ``constraint(model)`` after the variables and the winner
        constraints exist, so it can add ``model.model += ...`` freely.
        """
        self.extra_constraints.append(constraint)

    def build(self) -> None:
        """Create the variables, the constraints and the objective.

        Safe to call again after changing ``winners``: everything is rebuilt from
        scratch, and the targets are re-checked.
        """
        self._check_targets()
        rankings = self.rankings
        self.model = LpProblem("ranking_lp", LpMinimize)
        # Variables are named by position: a tuple-derived name can collide once
        # PuLP sanitises it, and CBC rejects duplicate names in one problem.
        self.variables = {
            ranking: LpVariable(f"x{i}", lowBound=0, cat="Integer")
            for i, ranking in enumerate(rankings)
        }
        counts = list(self.variables.values())

        for key, winner in self.winners.items():
            self._add_scoring_constraints(rule_for_key(key), winner)
        if self.condorcet_winner is not None:
            self._add_condorcet_constraints(self.condorcet_winner)

        if self.n_voters is not None:
            self.model += lpSum(counts) == self.n_voters
        else:
            self.model += lpSum(counts) <= self.max_added

        for constraint in self.extra_constraints:
            constraint(self)
        self._add_objective(counts)

    def _add_scoring_constraints(self, rule: ScoringRule, winner: int) -> None:
        """Require ``winner`` to out-score everyone under one scoring rule."""
        weights = np.asarray(rule.weights(self.n_candidates))

        # points[ranking][j] is what one voter with that ranking gives candidate j.
        def points(ranking: Ranking, candidate: int) -> float:
            return float(weights[ranking.index(candidate)])

        base = np.zeros(self.n_candidates)
        if self.base_profile is not None:
            base = np.asarray(rule.scores(self.base_profile), dtype=float)

        score = [
            base[j]
            + lpSum(
                self.variables[ranking] * points(ranking, j)
                for ranking in self.rankings
            )
            for j in range(self.n_candidates)
        ]
        for j in range(self.n_candidates):
            if j != winner:
                self.model += score[winner] >= score[j] + 1

    def _add_condorcet_constraints(self, winner: int) -> None:
        """Require ``winner`` to win every head-to-head duel.

        A duel is linear in the counts: each ranking contributes +1 to whichever
        of the two it puts first, so the margin is a plain weighted sum.
        """
        base_margins = (
            self.base_profile.margins
            if self.base_profile is not None
            else np.zeros((self.n_candidates, self.n_candidates), dtype=int)
        )
        for other in range(self.n_candidates):
            if other == winner:
                continue
            margin = int(base_margins[winner][other]) + lpSum(
                self.variables[ranking]
                * (1 if ranking.index(winner) < ranking.index(other) else -1)
                for ranking in self.rankings
            )
            self.model += margin >= 1

    def _add_objective(self, counts: list) -> None:
        """Set the configured objective over the per-ranking counts.

        The evenness objectives only shape *which* feasible electorate comes
        back; ``min_total`` changes its size instead. The auxiliary bounds are
        declared integer because the counts are, which lets the solver close the
        gap faster without changing the solution set.
        """
        if self.objective == "feasibility":
            return

        if self.objective == "min_total":
            self.model += lpSum(counts)

        elif self.objective == "minmax":
            largest = LpVariable("largest", lowBound=0, cat="Integer")
            self.model += largest
            for count in counts:
                self.model += count <= largest

        elif self.objective == "maxmin":
            smallest = LpVariable("smallest", lowBound=0, cat="Integer")
            self.model += -smallest  # minimising -m maximises m
            for count in counts:
                self.model += count >= smallest

        elif self.objective == "range":
            largest = LpVariable("largest", lowBound=0, cat="Integer")
            smallest = LpVariable("smallest", lowBound=0, cat="Integer")
            self.model += largest - smallest
            for count in counts:
                self.model += count <= largest
                self.model += count >= smallest

    # Solving:

    def solve(self, msg: bool = False) -> str:
        """Build if needed, solve, and return the solver status."""
        if self.model is None:
            self.build()
        self.model.solve(PULP_CBC_CMD(msg=msg))
        self.status = LpStatus[self.model.status]
        return self.status

    def solution_counts(self) -> dict:
        """``{ranking: how many voters the solver put there}``."""
        if self.variables is None:
            raise ValueError("model has not been built yet")
        return {
            ranking: int(variable.varValue or 0)
            for ranking, variable in self.variables.items()
            if (variable.varValue or 0) > 0
        }

    def solution_profile(self, include_base: bool = True) -> Profile:
        """The solution as a :class:`~voting.profile.Profile`.

        This is the honest way to check a solution: the model reasons in ranking
        counts, so reading it back as ranking counts introduces no sampling
        error. ``include_base`` folds in the fixed electorate of a swap model.
        """
        profile = Profile.from_counts(self.solution_counts(), self.n_candidates)
        if include_base and self.base_profile is not None:
            profile = profile + self.base_profile
        return profile

    def check(self) -> dict:
        """``{rule key: who actually wins}`` on the solved profile.

        Every target should appear with the candidate that was asked for.
        """
        profile = self.solution_profile()
        outcome = {key: rule_for_key(key).winner(profile) for key in self.winners}
        if self.condorcet_winner is not None:
            outcome["condorcet"] = profile.condorcet_winner()
        return outcome

    def generate_voter_positions(self) -> np.ndarray:
        """``(n, 2)`` voter positions realizing the solution.

        Each ranking's voters are drawn from the sampled points of its region, so
        they really do stand where that ranking holds.
        """
        chunks = []
        for ranking, count in self.solution_counts().items():
            points = self.regions[ranking]
            chosen = self.rng.choice(
                len(points), size=count, replace=count > len(points)
            )
            chunks.append(points[chosen])
        return np.vstack(chunks) if chunks else np.empty((0, 2))


# Cuts:

def exclude_current_solution(
    model: LpModel, min_new_voters: int = 1
) -> Callable[[LpModel], None]:
    """Forbid the regions the current solution left empty from staying empty.

    Add the result with ``model.add_constraint(...)`` and ``model.build()``
    again to get a structurally different placement. ``min_new_voters`` says how
    different: with 1, a single voter moving into an unused region is enough.

    Call this *after* ``solve()``, while the solved values are still on the
    variables - ``build()`` replaces them with fresh, unsolved ones.
    """
    unused = [r for r, v in model.variables.items() if not (v.varValue or 0)]
    if not unused:
        raise ValueError("every realizable ranking is already used; no cut to add")

    def cut(m: LpModel) -> None:
        m.model += lpSum(m.variables[r] for r in unused) >= min_new_voters

    return cut


def exclude_largest_variable(model: LpModel) -> Callable[[LpModel], None]:
    """Force the currently most crowded region to be empty next time."""
    largest = max(model.variables, key=lambda r: model.variables[r].varValue or 0)

    def cut(m: LpModel) -> None:
        m.model += m.variables[largest] == 0

    return cut


def exclude_current_solution_bigm(
    model: LpModel, big_m: int | None = None
) -> Callable[[LpModel], None]:
    """Forbid the exact solution just found, in every region at once.

    Where :func:`exclude_current_solution` only demands that some unused region
    fills up, this demands that *some* count changes by at least one, up or down.
    Two binary flags per ranking gate a pair of big-M constraints, and a final
    constraint forces at least one flag on.
    """
    target = {r: (v.varValue or 0) for r, v in model.variables.items()}
    limit = big_m if big_m is not None else (model.n_voters or model.max_added) + 1
    # Tag the flags with this cut's position so repeated cuts do not collide on
    # variable names - CBC rejects duplicates within one problem.
    tag = len(model.extra_constraints)

    def cut(m: LpModel) -> None:
        flags = []
        for i, (ranking, value) in enumerate(target.items()):
            below = LpVariable(f"below{tag}_{i}", cat="Binary")
            above = LpVariable(f"above{tag}_{i}", cat="Binary")
            m.model += m.variables[ranking] <= value - 1 + limit * (1 - below)
            m.model += m.variables[ranking] >= value + 1 - limit * (1 - above)
            flags += [below, above]
        m.model += lpSum(flags) >= 1

    return cut
