from abc import ABC, abstractmethod
from itertools import permutations
from typing import Callable

import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, PULP_CBC_CMD
from scipy.optimize import linprog

from candidate.candidate import Candidate
from voter.voter import Voter

Ranking = tuple[int, ...]

# Linear objectives that pick *which* feasible voter distribution to return.
# The first four are "evenness" objectives (they only shape the solution, not whether
# the winner constraints are satisfiable); "min_total" instead minimizes how many
# voters are used. In the swap model the total is always capped at n_voters, so:
#   "feasibility" - no objective, any feasible point (up to n_voters added)
#   "minmax"      - min M  s.t. x_s <= M           (cap the largest bucket)
#   "maxmin"      - max m  s.t. x_s >= m            (fill ~n_voters, evenly)
#   "range"       - min M - m  s.t. m <= x_s <= M   (squeeze the spread)
#   "min_total"   - min sum x                       (add as few voters as possible)
OBJECTIVES = ("feasibility", "minmax", "maxmin", "range", "min_total")


class LpModel(ABC):
    """Base class for LP models that find voter distributions producing target strategy winners."""

    def __init__(
        self,
        candidates: list,
        n_voters: int,
        winners: dict[str, int],
        bounds: tuple[float, float] = (-10.0, 10.0),
        rng: np.random.Generator | None = None,
        objective: str = "feasibility",
    ):
        if objective not in OBJECTIVES:
            raise ValueError(
                f"objective must be one of {OBJECTIVES}, got {objective!r}"
            )

        self.candidates = candidates
        self.n_voters = n_voters
        self.winners = winners
        self.bounds = bounds
        self.rng = rng if rng is not None else np.random.default_rng()
        self.objective = objective

        self.n_candidates = len(candidates)
        self.candidate_positions = np.array([c.position for c in candidates])

        self.model: LpProblem | None = None
        self.status: str | None = None

        self._points_by_ranking: dict[Ranking, list[np.ndarray]] | None = None

        # Extra constraints applied on top of the winner constraints in build().
        # Each callable receives the model itself and adds constraints via
        # `model.model += ...`, using `model.variables` (x_sigma per ranking).
        self.extra_constraints: list[Callable[["LpModel"], None]] = []

    def add_constraint(self, constraint: Callable[["LpModel"], None]) -> None:
        """Register an extra constraint, applied on the next build().

        `constraint` is called as `constraint(self)` from build(), after the
        winner constraints and variables are set up, so it can freely add
        `self.model += ...` expressions over `self.variables`.
        """
        self.extra_constraints.append(constraint)

    def _apply_extra_constraints(self) -> None:
        for constraint in self.extra_constraints:
            constraint(self)

    def _sample_pool(self, size: int = 200_000) -> dict[Ranking, list[np.ndarray]]:
        if self._points_by_ranking is not None:
            return self._points_by_ranking

        pool = self.rng.uniform(self.bounds[0], self.bounds[1], size=(size, 2))
        dists = np.linalg.norm(self.candidate_positions[None] - pool[:, None], axis=2)
        rankings = np.argsort(dists, axis=1)

        grouped: dict[Ranking, list[np.ndarray]] = {}
        for k in range(size):
            key = tuple(rankings[k].tolist())
            grouped.setdefault(key, []).append(pool[k])

        self._points_by_ranking = grouped
        return grouped

    def count_voters_by_ranking(self, voters: list) -> dict[Ranking, int]:
        """Group voters by the candidate ranking their position induces.

        Each voter is assigned the ranking of candidates sorted by Euclidean
        distance to the voter (closest first), exactly as in _sample_pool.

        voters : objects with a .position attribute

        Returns a mapping ranking -> number of voters falling in that region.
        """
        if not voters:
            return {}

        positions = np.array([v.position for v in voters])
        dists = np.linalg.norm(
            self.candidate_positions[None] - positions[:, None], axis=2
        )
        rankings = np.argsort(dists, axis=1)

        counts: dict[Ranking, int] = {}
        for k in range(len(voters)):
            key = tuple(rankings[k].tolist())
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _add_objective(self, variables: list[LpVariable]) -> None:
        """Set the configured objective (self.objective) over the count variables.

        The evenness objectives ("minmax"/"maxmin"/"range") only shape *which*
        feasible distribution is returned; "min_total" instead minimizes how many
        voters are used (size, not shape).

        Auxiliary variables (M, m) are declared Integer. They are functionally pinned
        to the (integer) x_s at any feasible point, so integrality doesn't change the
        solution set but it tightens the LP relaxation and lets CBC close the gap
        faster (notably for "range").
        """
        obj = self.objective
        if obj == "feasibility":
            return

        if obj == "min_total":
            self.model += lpSum(variables)

        elif obj == "minmax":
            M = LpVariable("M", lowBound=0, cat="Integer")
            self.model += M
            for v in variables:
                self.model += v <= M

        elif obj == "maxmin":
            m = LpVariable("m", lowBound=0, cat="Integer")
            self.model += -m  # LpMinimize, so minimizing -m maximizes m
            for v in variables:
                self.model += v >= m

        elif obj == "range":
            M = LpVariable("M", lowBound=0, cat="Integer")
            m = LpVariable("m", lowBound=0, cat="Integer")
            self.model += M - m
            for v in variables:
                self.model += v <= M
                self.model += v >= m

    @abstractmethod
    def build(self) -> None: ...

    def solve(self, msg: bool = False) -> str:
        if self.model is None:
            self.build()
        self.model.solve(PULP_CBC_CMD(msg=msg))
        self.status = LpStatus[self.model.status]
        return self.status

    def print_variables(self) -> None:
        if self.model is None:
            print("Model not built yet.")
            return
        for var in self.model.variables():
            print(f"{var.name}: {var.varValue}")

    @abstractmethod
    def generate_voter_positions(self) -> np.ndarray: ...


class PermutationLpModel(LpModel):
    """LP with x[sigma] = number of voters whose full ranking is sigma."""

    def __init__(
        self,
        candidates: list,
        n_voters: int,
        winners: dict[str, int],
        bounds: tuple[float, float] = (-10.0, 10.0),
        rng: np.random.Generator | None = None,
        pool_size: int = 300_000,
        objective: str = "feasibility",
    ):
        super().__init__(candidates, n_voters, winners, bounds, rng, objective)
        self.pool_size = pool_size
        self.variables: dict[Ranking, LpVariable] | None = None
        self.realizable_rankings: list[Ranking] | None = None

    def build(self) -> None:
        N = self.n_candidates
        V = self.n_voters

        pool = self._sample_pool(size=self.pool_size)
        self.realizable_rankings = list(pool.keys())

        self.model = LpProblem("permutation_lp", LpMinimize)
        self.variables = {
            sigma: LpVariable(f"x_{sigma}", lowBound=0, cat="Integer")
            for sigma in self.realizable_rankings
        }
        x = self.variables

        self.model += lpSum(x.values()) == V

        w_plur = self.winners.get("plurality")
        w_borda = self.winners.get("borda")
        w_veto = self.winners.get("veto")

        if w_plur is not None:
            plur_score = {
                j: lpSum(x[s] for s in self.realizable_rankings if s[0] == j)
                for j in range(N)
            }
            for j in range(N):
                if j != w_plur:
                    self.model += plur_score[w_plur] >= plur_score[j] + 1

        if w_borda is not None:
            borda_score = {
                j: lpSum(x[s] * (N - 1 - s.index(j)) for s in self.realizable_rankings)
                for j in range(N)
            }
            for j in range(N):
                if j != w_borda:
                    self.model += borda_score[w_borda] >= borda_score[j] + 1

        if w_veto is not None:
            last_count = {
                j: lpSum(x[s] for s in self.realizable_rankings if s[-1] == j)
                for j in range(N)
            }
            for j in range(N):
                if j != w_veto:
                    self.model += last_count[w_veto] + 1 <= last_count[j]

        self._apply_extra_constraints()
        self._add_objective(list(x.values()))

    def generate_voter_positions(self) -> np.ndarray:
        pool = self._sample_pool()
        positions = []
        for sigma in self.realizable_rankings:
            count = int(self.variables[sigma].varValue or 0)
            if count == 0:
                continue
            pool_pts = pool[sigma]
            indices = self.rng.choice(
                len(pool_pts), size=count, replace=(count > len(pool_pts))
            )
            for idx in indices:
                positions.append(pool_pts[idx])
        return np.array(positions)

    def print_variables(self) -> None:
        if self.variables is None:
            print("Model not built yet.")
            return
        for sigma, var in self.variables.items():
            print(f"{sigma}: {var.varValue}")


def exclude_current_solution(
    model: "LpModel", min_new_voters: int = 1
) -> Callable[["LpModel"], None]:
    """Capture `model`'s current solution and return a cut excluding it.

    Pass the result to `model.add_constraint(...)` and `model.build()` again
    to re-solve under "give me a *different* solution": rankings (regions)
    that were unused (x_sigma == 0) in the current solution must now hold at
    least `min_new_voters` voters in total. This forces a structurally
    different voter placement on every iteration.

    `min_new_voters` controls *how* different the next solution must be -
    with 1 (the default) a single voter moving into a previously-unused
    region is enough, which can leave the rest of the distribution almost
    identical. Raise it (e.g. to a fraction of `n_voters` /
    `max_added_voters`) to require a bigger shift away from the current
    solution.

    Must be called *after* `model.solve()`, while `model.variables` still
    holds the solved `varValue`s (the cut captures those values immediately;
    `build()` recreates `self.variables` with fresh, unsolved LpVariables).
    """
    unused = [
        sigma for sigma, var in model.variables.items() if not (var.varValue or 0)
    ]
    if not unused:
        raise ValueError("every realizable ranking is already used; no cut to add")

    def cut(m: "LpModel") -> None:
        m.model += lpSum(m.variables[sigma] for sigma in unused) >= min_new_voters

    return cut


def exclude_largest_variable(model: "LpModel") -> Callable[["LpModel"], None]:
    """Capture `model`'s current solution and return a cut that forces the
    variable with the largest current value to zero in the next solve.

    Must be called *after* `model.solve()`, while `model.variables` still
    holds the solved `varValue`s.
    """
    largest_sigma = max(
        model.variables,
        key=lambda sigma: model.variables[sigma].varValue or 0,
    )

    def cut(m: "LpModel") -> None:
        m.model += m.variables[largest_sigma] == 0

    return cut


def exclude_current_solution_bigm(
    model: "LpModel", big_m: int | None = None
) -> Callable[["LpModel"], None]:
    """Capture `model`'s current solution x* and return a cut that excludes it
    via a big-M disjunction, forcing the *next* solution to differ from x* in
    at least one variable (by at least 1, up or down).

    For every ranking sigma, two binary flags below_sigma/above_sigma gate a
    pair of constraints:
        x_sigma <= x*_sigma - 1 + M * (1 - below_sigma)
        x_sigma >= x*_sigma + 1 - M * (1 - above_sigma)
    When a flag is 0 the corresponding constraint is relaxed by M and becomes
    non-binding; when it's 1, x_sigma is pushed below (resp. above) x*_sigma.
    A global cut sum(below_sigma + above_sigma) >= 1 then forces at least one
    flag on, i.e. at least one x_sigma must move away from x*_sigma.

    `big_m` defaults to `model.n_voters + 1`, large enough that any x_sigma in
    [0, n_voters] satisfies a relaxed constraint regardless of x*_sigma.

    Must be called *after* `model.solve()`, while `model.variables` still
    holds the solved `varValue`s (the cut captures those values immediately;
    `build()` recreates `self.variables` with fresh, unsolved LpVariables).
    """
    x_star = {sigma: var.varValue or 0 for sigma, var in model.variables.items()}
    M = big_m if big_m is not None else model.n_voters + 1
    # Tag the auxiliary binaries with the cut's position in extra_constraints so
    # repeated cuts (one per enumeration round) don't collide on variable names -
    # every cut iterates over the same set of rankings, and PuLP/CBC chokes on
    # duplicate variable names across constraints in the same model.
    tag = len(model.extra_constraints)

    def cut(m: "LpModel") -> None:
        flags = []
        for sigma, target in x_star.items():
            below = LpVariable(f"below_{tag}_{sigma}", cat="Binary")
            above = LpVariable(f"above_{tag}_{sigma}", cat="Binary")
            m.model += m.variables[sigma] <= target - 1 + M * (1 - below)
            m.model += m.variables[sigma] >= target + 1 - M * (1 - above)
            flags.append(below)
            flags.append(above)
        m.model += lpSum(flags) >= 1

    return cut


class PermutationSwapLpModel(LpModel):
    """LP with x[sigma] = number of voters whose full ranking is sigma."""

    def __init__(
        self,
        candidates: list[Candidate],
        voters: list[Voter],
        max_added_voters: int,
        winners: dict[str, int],
        bounds: tuple[float, float] = (-10.0, 10.0),
        rng: np.random.Generator | None = None,
        pool_size: int = 300_000,
        objective: str = "feasibility",
    ):
        super().__init__(candidates, max_added_voters, winners, bounds, rng, objective)
        self.pool_size = pool_size
        self.variables: dict[Ranking, LpVariable] | None = None
        self.realizable_rankings: list[Ranking] | None = None
        self.voters = voters
        self.max_added_voters = max_added_voters

    def build(self) -> None:
        N = self.n_candidates

        pool = self._sample_pool(size=self.pool_size)
        self.realizable_rankings = list(pool.keys())

        counter = self.count_voters_by_ranking(self.voters)

        self.model = LpProblem("permutation_lp", LpMinimize)
        self.variables = {
            sigma: LpVariable(f"x_{sigma}", lowBound=0, cat="Integer")
            for sigma in self.realizable_rankings
        }
        x = self.variables

        w_plur = self.winners.get("plurality")
        w_borda = self.winners.get("borda")
        w_veto = self.winners.get("veto")

        if w_plur is not None:
            plur_count = {
                j: sum(count for s, count in counter.items() if s[0] == j)
                for j in range(N)
            }
            plur_score = {
                j: plur_count[j]
                + lpSum(x[s] for s in self.realizable_rankings if s[0] == j)
                for j in range(N)
            }
            for j in range(N):
                if j != w_plur:
                    self.model += plur_score[w_plur] >= plur_score[j] + 1

        if w_borda is not None:
            borda_count = {
                j: sum(count * (N - 1 - s.index(j)) for s, count in counter.items())
                for j in range(N)
            }
            borda_score = {
                j: borda_count[j]
                + lpSum(x[s] * (N - 1 - s.index(j)) for s in self.realizable_rankings)
                for j in range(N)
            }
            for j in range(N):
                if j != w_borda:
                    self.model += borda_score[w_borda] >= borda_score[j] + 1

        if w_veto is not None:
            veto_count = {
                j: sum(count for s, count in counter.items() if s[-1] == j)
                for j in range(N)
            }
            last_count = {
                j: veto_count[j]
                + lpSum(x[s] for s in self.realizable_rankings if s[-1] == j)
                for j in range(N)
            }
            for j in range(N):
                if j != w_veto:
                    self.model += last_count[w_veto] + 1 <= last_count[j]

        # Cap how many voters we may add; `objective` decides how to use that budget
        # ("min_total" => as few as possible, "minmax"/"range" => few and spread,
        # "maxmin" => fill ~max_added_voters evenly, "feasibility" => anything).
        self.model += lpSum(x.values()) <= self.max_added_voters
        self._apply_extra_constraints()
        self._add_objective(list(x.values()))

    def generate_voter_positions(self) -> np.ndarray:
        pool = self._sample_pool()
        positions = []
        for sigma in self.realizable_rankings:
            count = int(self.variables[sigma].varValue or 0)
            if count == 0:
                continue
            pool_pts = pool[sigma]
            indices = self.rng.choice(
                len(pool_pts), size=count, replace=(count > len(pool_pts))
            )
            for idx in indices:
                positions.append(pool_pts[idx])
        return np.array(positions)

    def print_variables(self) -> None:
        if self.variables is None:
            print("Model not built yet.")
            return
        for sigma, var in self.variables.items():
            print(f"{sigma}: {var.varValue}")
