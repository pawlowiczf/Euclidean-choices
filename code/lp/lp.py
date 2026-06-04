from abc import ABC, abstractmethod
from itertools import permutations

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
