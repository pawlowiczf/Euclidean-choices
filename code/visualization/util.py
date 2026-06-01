import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import numpy as np
import random

from election.result import ElectionResult


def random_2d_point(bounds_x=(-5, 5), bounds_y=(-5, 5)) -> tuple[float, float]:
    x = random.uniform(bounds_x[0], bounds_x[1])
    y = random.uniform(bounds_y[0], bounds_y[1])
    return (x, y)


def random_2d_points(n: int = 5, bounds_x=(-5, 5), bounds_y=(-5, 5)) -> np.ndarray:
    points = np.zeros((n, 2))

    for i in range(n):
        x = random.uniform(bounds_x[0], bounds_x[1])
        y = random.uniform(bounds_y[0], bounds_y[1])
        points[i] = (x, y)

    return points


def plot(candidates: np.ndarray, voters: np.ndarray, mpl_params: dict = None):
    plt.scatter(
        candidates[:, 0],
        candidates[:, 1],
        color="green",
        marker="x",
        label="Candidates",
    )
    plt.scatter(voters[:, 0], voters[:, 1], marker="o", label="Voters")


def plot_results(result: ElectionResult):
    candidates_arr = np.array([c.position for c in result.candidates])
    voters_arr = np.array([v.position for v in result.voters])

    plt.scatter(
        candidates_arr[:, 0],
        candidates_arr[:, 1],
        color="green",
        marker="x",
        label="Candidates",
    )
    plt.scatter(
        voters_arr[:, 0],
        voters_arr[:, 1],
        marker="o",
        color="gray",
        label="Voters",
        alpha=0.4,
    )

    winner_to_strategies = {}
    for strategy_name, winner in result.winners().items():
        winner_to_strategies.setdefault(winner.id, (winner, []))[1].append(
            strategy_name
        )

    colors = plt.cm.tab10.colors
    for i, (winner, strategy_names) in enumerate(winner_to_strategies.values()):
        label = "Winner:\n" + "\n".join(strategy_names)
        plt.scatter(*winner.position, marker="*", s=300, color=colors[i], label=label)

    plt.gcf().set_size_inches(8, 7)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_lp_result(
    candidates: list,
    positions: np.ndarray,
    winners: dict[str, int] | None = None,
    bounds: tuple[float, float] = (-5.0, 5.0),
    ax: plt.Axes | None = None,
):
    """Plot voter positions from an LpModel: color voters by top choice, ring strategy winners.

    candidates : list of objects with a .position attribute
    positions  : (N, 2) array of voter positions; rows containing NaN are skipped
    winners    : optional, e.g. {"plurality": 0, "borda": 1, "veto": 2}
    bounds     : (lo, hi) — sets fixed xlim and ylim for stable plot scale across runs
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(9, 8))

    candidate_positions = np.array([c.position for c in candidates])
    n_candidates = len(candidates)

    valid = ~np.isnan(positions[:, 0])
    placed = positions[valid]

    dists = np.linalg.norm(candidate_positions[None] - placed[:, None], axis=2)
    top_choice = np.argmin(dists, axis=1)

    colors = plt.cm.tab10.colors
    for cand_idx in range(n_candidates):
        mask = top_choice == cand_idx
        if not mask.any():
            continue
        pts = placed[mask]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            color=colors[cand_idx % len(colors)],
            s=60,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
            label=f"Top choice: C{cand_idx} ({int(mask.sum())} voters)",
        )

    for idx, c in enumerate(candidates):
        ax.scatter(*c.position, marker="x", s=150, c=[colors[idx % len(colors)]], linewidths=3, zorder=4)
        ax.annotate(
            f"C{idx}",
            c.position,
            fontsize=12,
            fontweight="bold",
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
        )

    winner_proxies = []
    if winners:
        for strategy, idx in winners.items():
            ring_color = colors[idx % len(colors)]
            ring_size = 350
            ax.scatter(
                *candidates[idx].position,
                marker="o",
                s=ring_size,
                facecolors="none",
                edgecolors=ring_color,
                linewidths=3,
                zorder=5,
            )
            winner_proxies.append(
                mlines.Line2D(
                    [], [],
                    marker="o",
                    linestyle="None",
                    markerfacecolor="none",
                    markeredgecolor=ring_color,
                    markersize=10,
                    markeredgewidth=2,
                    label=f"{strategy} → C{idx}",
                )
            )

    n_skipped = int((~valid).sum())
    title = f"Voter Distribution in Euclidean Space (n = {len(placed)})"
    if n_skipped:
        title += f"\n{n_skipped} voters excluded: ranking not realizable in 2D"

    margin = 0.5
    ax.set_xlim(bounds[0] - margin, bounds[1] + margin)
    ax.set_ylim(bounds[0] - margin, bounds[1] + margin)
    ax.set_aspect("equal")
    ax.set_title(title)
    voter_handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=voter_handles + winner_proxies,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncols=3,
        fontsize=8,
    )

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_winner_distance_histogram(
    distances: list[float], bins: int = 5, ax: plt.Axes | None = None
):
    standalone = ax is None
    if standalone:
        ax = plt.gca()

    ax.hist(distances, bins=bins, edgecolor="black")
    ax.set_xlabel("Average distance between winners")
    ax.set_ylabel("Number of elections")
    ax.set_title("Distribution of winner dispersion across strategies")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    if standalone:
        plt.show()
