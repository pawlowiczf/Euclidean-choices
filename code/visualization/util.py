import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import numpy as np
import random

from election.result import ElectionResult
from strategy.plurality import PluralityStrategy
from strategy.borda import BordaCountStrategy
from strategy.veto import VetoStrategy

# Map a strategy key (as used in the `winners` dict) to its human-readable name,
# so legends read "Borda count" instead of "borda".
_STRATEGY_NAMES = {
    s.key: s.name for s in (PluralityStrategy(), BordaCountStrategy(), VetoStrategy())
}


def _strategy_label(key: str) -> str:
    """Human-readable strategy name for a winners-dict key, falling back to the key."""
    return _STRATEGY_NAMES.get(key, key)


def random_2d_point(bounds_x=(-10, 10), bounds_y=(-10, 10)) -> tuple[float, float]:
    x = random.uniform(bounds_x[0], bounds_x[1])
    y = random.uniform(bounds_y[0], bounds_y[1])
    return (x, y)


def random_2d_points(n: int = 5, bounds_x=(-10, 10), bounds_y=(-10, 10)) -> np.ndarray:
    points = np.zeros((n, 2))

    for i in range(n):
        x = random.uniform(bounds_x[0], bounds_x[1])
        y = random.uniform(bounds_y[0], bounds_y[1])
        points[i] = (x, y)

    return points


def _strip_axes(ax: plt.Axes):
    """Remove tick marks and tick labels (the numbers on the axes) while keeping
    the box/frame around the plot."""
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        labelbottom=False,
        labelleft=False,
    )


def plot(candidates: np.ndarray, voters: np.ndarray, mpl_params: dict = None):
    plt.scatter(
        candidates[:, 0],
        candidates[:, 1],
        color="green",
        marker="x",
        label="Candidates",
    )
    plt.scatter(voters[:, 0], voters[:, 1], marker="o", label="Voters")


def plot_results(
    result: ElectionResult,
    bounds: tuple[float, float] = (-10.0, 10.0),
    ax: plt.Axes | None = None,
):
    """Plot an ElectionResult in the same plain style as plot_lp_swap_result: gray
    voter dots, dark candidate dots, colored winner dots, a bottom legend and
    stripped axes on a fixed square scale."""
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(9, 8))

    voters_arr = np.array([v.position for v in result.voters])

    winners_dict = result.winners()
    winner_ids = {w.id for w in winners_dict.values()}

    ax.scatter(
        voters_arr[:, 0],
        voters_arr[:, 1],
        marker="o",
        color="gray",
        s=45,
        alpha=0.4,
        label=f"Voters ({len(voters_arr)})",
    )
    # Non-winning candidates are dark dots; a winner's marker becomes a colored dot
    # (drawn below) so the two symbols never stack on the same point.
    cand_label_done = False
    for c in result.candidates:
        if c.id not in winner_ids:
            ax.scatter(
                *c.position,
                marker="o",
                s=90,
                c="#555555",
                zorder=4,
                label=None if cand_label_done else "Candidate",
            )
            cand_label_done = True
        ax.annotate(
            f"C{c.id}",
            c.position,
            fontsize=12,
            fontweight="bold",
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
        )

    colors = plt.cm.tab10.colors
    winner_proxies = []
    winner_level = {}
    for s_idx, (strategy_name, winner) in enumerate(winners_dict.items()):
        level = winner_level.get(winner.id, 0)
        winner_level[winner.id] = level + 1
        win_color = colors[s_idx % len(colors)]
        ax.scatter(
            *winner.position,
            marker="o",
            s=160 + level * 180,
            facecolors=win_color,
            edgecolors="black",
            linewidths=1.2,
            zorder=5 - level * 0.1,
        )
        winner_proxies.append(
            mlines.Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor=win_color,
                markeredgecolor="black",
                markersize=11,
                markeredgewidth=1.2,
                label=f"Winner – {_strategy_label(strategy_name)} (C{winner.id})",
            )
        )

    title = f"Voter Distribution in Euclidean Space ({len(voters_arr)} voters)"
    margin = 0.5
    ax.set_xlim(bounds[0] - margin, bounds[1] + margin)
    ax.set_ylim(bounds[0] - margin, bounds[1] + margin)
    ax.set_aspect("equal")
    ax.set_title(title)
    _strip_axes(ax)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + winner_proxies,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncols=3,
        fontsize=8,
        labelspacing=0.9,
        columnspacing=1.5,
        handletextpad=0.6,
    )

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_lp_result(
    candidates: list,
    positions: np.ndarray,
    winners: dict[str, int] | None = None,
    bounds: tuple[float, float] = (-10.0, 10.0),
    ax: plt.Axes | None = None,
    color_voters: bool = True,
):
    """Plot voter positions from an LpModel.

    candidates   : list of objects with a .position attribute
    positions    : (N, 2) array of voter positions; rows containing NaN are skipped
    winners      : optional, e.g. {"plurality": 0, "borda": 1, "veto": 2}
    bounds       : (lo, hi) — sets fixed xlim and ylim for stable plot scale across runs
    color_voters : if True, color voters by their top choice and draw candidates as
                   colored "x" / winners as stars. If False, use a plain style: gray
                   voter dots, black candidate dots, colored winner dots (all circles).
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(9, 8))

    candidate_positions = np.array([c.position for c in candidates])
    n_candidates = len(candidates)

    valid = ~np.isnan(positions[:, 0])
    placed = positions[valid]

    colors = plt.cm.tab10.colors
    winner_idxs = set(winners.values()) if winners else set()

    if color_voters:
        dists = np.linalg.norm(candidate_positions[None] - placed[:, None], axis=2)
        top_choice = np.argmin(dists, axis=1)
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
    else:
        ax.scatter(
            placed[:, 0],
            placed[:, 1],
            color="gray",
            s=45,
            alpha=0.4,
            label=f"Voters ({len(placed)})",
        )

    # Candidate markers. In colored mode non-winners are "x"; a winner's marker
    # becomes a star (drawn below) so the symbols never stack. In plain mode every
    # candidate is a dot, black for non-winners and colored for winners.
    cand_label_done = False
    for idx, c in enumerate(candidates):
        if idx not in winner_idxs:
            if color_voters:
                ax.scatter(
                    *c.position,
                    marker="x",
                    s=150,
                    c=[colors[idx % len(colors)]],
                    linewidths=3,
                    zorder=4,
                )
            else:
                ax.scatter(
                    *c.position,
                    marker="o",
                    s=90,
                    c="#555555",
                    zorder=4,
                    label=None if cand_label_done else "Candidate",
                )
                cand_label_done = True
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
        # Colored mode: a star filled in the candidate's color so it stands out
        # inside a same-colored cluster. Plain mode: a dot in a per-strategy color
        # so winners pop against the gray voters / black candidates. Either way,
        # nest the marks (smaller on top) when strategies share a candidate.
        winner_level = {}
        for s_idx, (strategy, idx) in enumerate(winners.items()):
            level = winner_level.get(idx, 0)
            winner_level[idx] = level + 1
            if color_voters:
                marker = "*"
                win_color = colors[idx % len(colors)]
                marker_size = 320 + level * 320
                proxy_size = 11
            else:
                marker = "o"
                win_color = colors[s_idx % len(colors)]
                marker_size = 160 + level * 180
                proxy_size = 11
            ax.scatter(
                *candidates[idx].position,
                marker=marker,
                s=marker_size,
                facecolors=win_color,
                edgecolors="black",
                linewidths=1.2,
                zorder=5 - level * 0.1,
            )
            winner_proxies.append(
                mlines.Line2D(
                    [],
                    [],
                    marker=marker,
                    linestyle="None",
                    markerfacecolor=win_color,
                    markeredgecolor="black",
                    markersize=proxy_size,
                    markeredgewidth=1.2,
                    label=f"Winner – {_strategy_label(strategy)} (C{idx})",
                )
            )

    n_skipped = int((~valid).sum())
    title = f"Voter Distribution in Euclidean Space ({len(placed)} voters)"
    if n_skipped:
        title += f"\n{n_skipped} voters excluded: ranking not realizable in 2D"

    margin = 0.5
    ax.set_xlim(bounds[0] - margin, bounds[1] + margin)
    ax.set_ylim(bounds[0] - margin, bounds[1] + margin)
    ax.set_aspect("equal")
    ax.set_title(title)
    _strip_axes(ax)
    voter_handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=voter_handles + winner_proxies,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncols=3,
        fontsize=8,
        labelspacing=0.9,
        columnspacing=1.5,
        handletextpad=0.6,
    )

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_lp_swap_result(
    candidates: list,
    positions: np.ndarray,
    new_positions: np.ndarray,
    winners: dict[str, int] | None = None,
    bounds: tuple[float, float] = (-10.0, 10.0),
    ax: plt.Axes | None = None,
    legend: bool = True,
):
    """Plot a swap-LP result: the fixed voters plus the voters the LP added.

    Uses the plain style of plot_lp_result(color_voters=False): existing voters are
    gray dots, candidates are dark dots (winners colored), and the voters added by
    the swap model are drawn as green circles on top.

    candidates    : list of objects with a .position attribute
    positions     : (N, 2) array of the existing/fixed voter positions
    new_positions : (M, 2) array of voters added by the swap LP; NaN rows skipped
    winners       : optional, e.g. {"plurality": 0, "borda": 1, "veto": 2}
    bounds        : (lo, hi) — fixed xlim and ylim for stable plot scale across runs
    legend        : if False, skip the per-axes legend (e.g. when tiling subplots)
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(9, 8))

    colors = plt.cm.tab10.colors
    winner_idxs = set(winners.values()) if winners else set()

    existing = positions[~np.isnan(positions[:, 0])]
    added = (
        new_positions[~np.isnan(new_positions[:, 0])]
        if len(new_positions)
        else new_positions
    )

    ax.scatter(
        existing[:, 0],
        existing[:, 1],
        color="gray",
        s=45,
        alpha=0.4,
        label=f"Existing voters ({len(existing)})",
    )
    if len(added):
        ax.scatter(
            added[:, 0],
            added[:, 1],
            color="green",
            s=55,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
            label=f"Added voters ({len(added)})",
        )

    # Candidates as dark dots (a winner's marker is colored, drawn below).
    cand_label_done = False
    for idx, c in enumerate(candidates):
        if idx not in winner_idxs:
            ax.scatter(
                *c.position,
                marker="o",
                s=90,
                c="#555555",
                zorder=4,
                label=None if cand_label_done else "Candidate",
            )
            cand_label_done = True
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
        # Colored dot per strategy, nested (smaller on top) when strategies share a
        # candidate, so winners pop against the gray/green voters and dark candidates.
        winner_level = {}
        for s_idx, (strategy, idx) in enumerate(winners.items()):
            level = winner_level.get(idx, 0)
            winner_level[idx] = level + 1
            win_color = colors[s_idx % len(colors)]
            ax.scatter(
                *candidates[idx].position,
                marker="o",
                s=160 + level * 180,
                facecolors=win_color,
                edgecolors="black",
                linewidths=1.2,
                zorder=5 - level * 0.1,
            )
            winner_proxies.append(
                mlines.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=win_color,
                    markeredgecolor="black",
                    markersize=11,
                    markeredgewidth=1.2,
                    label=f"Winner – {_strategy_label(strategy)} (C{idx})",
                )
            )

    title = (
        "Voter Distribution in Euclidean Space "
        f"({len(existing) + len(added)} voters: {len(existing)} + {len(added)} added)"
    )
    margin = 0.5
    ax.set_xlim(bounds[0] - margin, bounds[1] + margin)
    ax.set_ylim(bounds[0] - margin, bounds[1] + margin)
    ax.set_aspect("equal")
    ax.set_title(title)
    _strip_axes(ax)
    if legend:
        voter_handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=voter_handles + winner_proxies,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            ncols=3,
            fontsize=8,
            labelspacing=0.9,
            columnspacing=1.5,
            handletextpad=0.6,
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
