from abc import ABC, abstractmethod
from itertools import permutations

import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpStatus, PULP_CBC_CMD
from scipy.optimize import linprog


Ranking = tuple[int, ...]


class LpModel(ABC):
    """Base class for LP models that find voter distributions producing target strategy winners."""

    def __init__(
        self,
        candidates: list,
        n_voters: int,
        winners: dict[str, int],
        bounds: tuple[float, float] = (-5.0, 5.0),
        rng: np.random.Generator | None = None,
    ):
        self.candidates = candidates
        self.n_voters = n_voters
        self.winners = winners
        self.bounds = bounds
        self.rng = rng if rng is not None else np.random.default_rng()

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

    @abstractmethod
    def build(self) -> None: ...

    def solve(self, msg: bool = False) -> str:
        if self.model is None:
            self.build()
        self.model.solve(PULP_CBC_CMD(msg=msg))
        self.status = LpStatus[self.model.status]
        return self.status

    @abstractmethod
    def generate_voter_positions(self) -> np.ndarray: ...


class MarginalLpModel(LpModel):
    """LP with c[i][r] = number of voters who have candidate i at rank r."""

    def __init__(
        self,
        candidates: list,
        n_voters: int,
        winners: dict[str, int],
        bounds: tuple[float, float] = (-5.0, 5.0),
        rng: np.random.Generator | None = None,
    ):
        super().__init__(candidates, n_voters, winners, bounds, rng)
        self.variables: list[list[LpVariable]] | None = None

    def build(self) -> None:
        N = self.n_candidates
        V = self.n_voters
        self.model = LpProblem("marginal_lp", LpMinimize)
        self.variables = [
            [LpVariable(f"c_{i}_{r}", lowBound=0, cat="Integer") for r in range(N)]
            for i in range(N)
        ]
        c = self.variables

        for r in range(N):
            self.model += lpSum(c[i][r] for i in range(N)) == V
        for i in range(N):
            self.model += lpSum(c[i][r] for r in range(N)) == V

        w_plur = self.winners.get("plurality")
        w_borda = self.winners.get("borda")
        w_veto = self.winners.get("veto")

        if w_plur is not None:
            for i in range(N):
                if i != w_plur:
                    self.model += c[w_plur][0] >= c[i][0] + 1

        if w_borda is not None:
            for i in range(N):
                if i == w_borda:
                    continue
                self.model += (
                    lpSum((N - 1 - r) * c[w_borda][r] for r in range(N))
                    >= lpSum((N - 1 - r) * c[i][r] for r in range(N)) + 1
                )

        if w_veto is not None:
            for i in range(N):
                if i == w_veto:
                    continue
                self.model += (
                    lpSum(c[w_veto][r] for r in range(N - 1))
                    >= lpSum(c[i][r] for r in range(N - 1)) + 1
                )

    def get_matrix(self) -> np.ndarray:
        N = self.n_candidates
        return np.array(
            [
                [int(self.variables[i][r].varValue or 0) for r in range(N)]
                for i in range(N)
            ]
        )

    def generate_voter_positions(self) -> np.ndarray:
        N = self.n_candidates
        V = self.n_voters
        remaining = self.get_matrix()

        voter_rankings: list[list[int]] = []
        for voter_idx in range(V):
            ranking = [-1] * N
            used: set[int] = set()

            def fill(r: int) -> bool:
                if r == N:
                    return True
                available = [
                    i for i in range(N) if remaining[i][r] > 0 and i not in used
                ]
                for k in self.rng.permutation(len(available)):
                    chosen = available[int(k)]
                    ranking[r] = chosen
                    used.add(chosen)
                    remaining[chosen][r] -= 1
                    if fill(r + 1):
                        return True
                    ranking[r] = -1
                    used.remove(chosen)
                    remaining[chosen][r] += 1
                return False

            if not fill(0):
                raise RuntimeError(
                    f"cannot build ranking for voter {voter_idx + 1}; "
                    f"matrix not decomposable (shouldn't happen for LP-produced matrices)"
                )
            voter_rankings.append(ranking.copy())

        pool = self._sample_pool()
        positions = []
        for ranking in voter_rankings:
            pts = pool.get(tuple(ranking), [])
            if not pts:
                positions.append(np.array([np.nan, np.nan]))
                continue
            positions.append(pts[self.rng.integers(0, len(pts))])
        return np.array(positions)


class PermutationLpModel(LpModel):
    """LP with x[sigma] = number of voters whose full ranking is sigma."""

    def __init__(
        self,
        candidates: list,
        n_voters: int,
        winners: dict[str, int],
        bounds: tuple[float, float] = (-5.0, 5.0),
        rng: np.random.Generator | None = None,
        pool_size: int = 300_000,
    ):
        super().__init__(candidates, n_voters, winners, bounds, rng)
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
            sigma: LpVariable(f"x_{idx}", lowBound=0, cat="Integer")
            for idx, sigma in enumerate(self.realizable_rankings)
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
