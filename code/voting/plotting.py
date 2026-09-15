"""Pictures of elections.

One drawing function does the work - :func:`plot_election` - and the rest are
short calls into it. The previous version had three near-identical 150-line
functions that each repeated the axes setup, the legend, the colour palette and
the winner-marker nesting; those live in one place here.

Two views are new and answer questions the scatter plot cannot:

:func:`plot_election` with ``regions=True`` shades the plurality regions, so you
see *why* a candidate wins - the picture of who is nearest where.

:func:`plot_majority_graph` draws the tournament, with an arrow from each winner
of a duel to its loser. Cycles are what make Condorcet interesting, and in this
view they are literally visible as loops.
"""

import random

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from voting.profile import Profile
from voting.rules import rule_name

PALETTE = plt.cm.tab10.colors
VOTER_GREY = "#808080"
CANDIDATE_DARK = "#555555"


# --------------------------------------------------------------------------
# Random points, kept from the original helpers
# --------------------------------------------------------------------------


def random_2d_point(bounds=(-10, 10)) -> np.ndarray:
    """One uniform point in the square ``bounds``."""
    return np.array([random.uniform(*bounds), random.uniform(*bounds)])


def random_2d_points(n: int = 5, bounds=(-10, 10)) -> np.ndarray:
    """``(n, 2)`` uniform points in the square ``bounds``."""
    return np.array([random_2d_point(bounds) for _ in range(n)])


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------


def _positions(items) -> np.ndarray:
    """``(n, 2)`` from Candidate/Voter objects or from bare coordinates."""
    items = list(items)
    if not items:
        return np.empty((0, 2))
    if hasattr(items[0], "position"):
        return np.array([item.position for item in items], dtype=float)
    return np.asarray(items, dtype=float)


def _proxy(marker: str, facecolor, label: str, size: float = 11):
    """A legend entry for something drawn with scatter."""
    return mlines.Line2D(
        [],
        [],
        marker=marker,
        linestyle="None",
        markerfacecolor=facecolor,
        markeredgecolor="black",
        markersize=size,
        markeredgewidth=1.2,
        label=label,
    )


def _label_candidate(ax, index: int, position, bounds) -> None:
    """Name a candidate, below the marker when it would run off the top edge."""
    near_top = position[1] > bounds[1] - 0.1 * (bounds[1] - bounds[0])
    offset, valign = ((0, -14), "top") if near_top else ((0, 14), "bottom")
    ax.annotate(
        f"C{index}",
        position,
        fontsize=12,
        fontweight="bold",
        xytext=offset,
        textcoords="offset points",
        ha="center",
        va=valign,
    )


def _finish(ax, bounds, title, extra_handles=(), legend=True, standalone=False):
    """The frame every plot in this module shares."""
    margin = 0.5
    ax.set_xlim(bounds[0] - margin, bounds[1] + margin)
    ax.set_ylim(bounds[0] - margin, bounds[1] + margin)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    ax.tick_params(
        axis="both", which="both", length=0, labelbottom=False, labelleft=False
    )
    if legend:
        handles, _ = ax.get_legend_handles_labels()
        handles = handles + list(extra_handles)
        if handles:
            ax.legend(
                handles=handles,
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


def _shade_regions(ax, candidates: np.ndarray, bounds, resolution: int = 400) -> None:
    """Tint the plane by which candidate is nearest - the plurality regions."""
    axis = np.linspace(bounds[0], bounds[1], resolution)
    grid_x, grid_y = np.meshgrid(axis, axis)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    nearest = np.argmin(
        np.linalg.norm(grid[:, None, :] - candidates[None, :, :], axis=2), axis=1
    )
    ax.imshow(
        nearest.reshape(resolution, resolution),
        origin="lower",
        extent=(bounds[0], bounds[1], bounds[0], bounds[1]),
        cmap=plt.cm.tab10,
        vmin=0,
        vmax=9,
        alpha=0.18,
        interpolation="nearest",
        zorder=0,
    )


def _draw_winners(ax, candidates: np.ndarray, winners: dict) -> list:
    """A coloured dot per rule, nested when several rules share a winner."""
    proxies, seen = [], {}
    for order, (key, index) in enumerate(winners.items()):
        level = seen.get(index, 0)
        seen[index] = level + 1
        colour = PALETTE[order % len(PALETTE)]
        ax.scatter(
            *candidates[index],
            marker="o",
            s=160 + level * 180,
            facecolors=colour,
            edgecolors="black",
            linewidths=1.2,
            zorder=5 - level * 0.1,
        )
        proxies.append(_proxy("o", colour, f"{rule_name(key)} → C{index}"))
    return proxies


# --------------------------------------------------------------------------
# The election picture
# --------------------------------------------------------------------------


def plot_election(
    candidates,
    voters=None,
    winners: dict | None = None,
    *,
    added_voters=None,
    color_by_choice: bool = False,
    regions: bool = False,
    bounds: tuple[float, float] = (-10.0, 10.0),
    title: str | None = None,
    ax=None,
    legend: bool = True,
):
    """Draw candidates, voters and - if given - who won under which rule.

    candidates      Candidate objects or an ``(M, 2)`` array.
    voters          Voter objects or an ``(N, 2)`` array; may be omitted.
    winners         ``{rule key: candidate index}``, e.g. ``result.winner_indices()``.
    added_voters    A second group drawn in green on top, for the LP models that
                    add voters to an existing electorate.
    color_by_choice Colour each voter by the candidate they rank first.
    regions         Shade the plane by nearest candidate (the plurality regions).
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(9, 8))

    candidate_positions = _positions(candidates)
    voter_positions = _positions(voters) if voters is not None else np.empty((0, 2))
    added_positions = (
        _positions(added_voters) if added_voters is not None else np.empty((0, 2))
    )
    winners = winners or {}

    if regions and len(candidate_positions):
        _shade_regions(ax, candidate_positions, bounds)

    if color_by_choice and len(voter_positions):
        choices = Profile.from_positions(
            voter_positions, candidate_positions
        ).top_choices()
        for index in range(len(candidate_positions)):
            group = voter_positions[choices == index]
            if len(group):
                ax.scatter(
                    group[:, 0],
                    group[:, 1],
                    color=PALETTE[index % len(PALETTE)],
                    s=55,
                    alpha=0.75,
                    edgecolors="black",
                    linewidths=0.4,
                    label=f"Prefers C{index} ({len(group)})",
                )
    elif len(voter_positions):
        ax.scatter(
            voter_positions[:, 0],
            voter_positions[:, 1],
            color=VOTER_GREY,
            s=45,
            alpha=0.45,
            label=f"Voters ({len(voter_positions)})",
        )

    if len(added_positions):
        ax.scatter(
            added_positions[:, 0],
            added_positions[:, 1],
            color="green",
            s=55,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
            label=f"Added voters ({len(added_positions)})",
        )

    winning = set(winners.values())
    labelled = False
    for index, position in enumerate(candidate_positions):
        if index not in winning:
            ax.scatter(
                *position,
                marker="o",
                s=90,
                color=CANDIDATE_DARK,
                zorder=4,
                label=None if labelled else "Candidates",
            )
            labelled = True
        _label_candidate(ax, index, position, bounds)

    proxies = _draw_winners(ax, candidate_positions, winners)

    if title is None:
        total = len(voter_positions) + len(added_positions)
        title = f"{len(candidate_positions)} candidates, {total} voters"
    _finish(ax, bounds, title, proxies, legend=legend, standalone=standalone)
    return ax


def plot_result(result, **kwargs):
    """Draw an :class:`~voting.election.ElectionResult`.

    A shortcut for :func:`plot_election` that unpacks the result's candidates,
    voters and winners.
    """
    kwargs.setdefault("title", result.label)
    return plot_election(
        result.candidates,
        result.voters,
        result.winner_indices(),
        **kwargs,
    )


# --------------------------------------------------------------------------
# The tournament
# --------------------------------------------------------------------------


def plot_majority_graph(
    profile: Profile,
    candidates=None,
    *,
    bounds: tuple[float, float] = (-10.0, 10.0),
    ax=None,
    title: str | None = None,
):
    """Draw the majority graph: an arrow from the winner of each duel to the loser.

    Candidates sit at their real positions when ``candidates`` is given, and on a
    circle otherwise. Arrow width grows with the margin. A Condorcet winner (all
    arrows outgoing) is ringed; if there is none, the cycle is there to be seen.
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(7, 7))

    n = profile.n_candidates
    if candidates is not None:
        points = _positions(candidates)
        span = bounds
    else:
        angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        span = (-1.4, 1.4)

    margins = profile.margins
    widest = max(margins.max(), 1)

    for a in range(n):
        for b in range(n):
            if margins[a, b] <= 0:
                continue
            start, end = points[a], points[b]
            direction = end - start
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            # Stop short of the node so the arrowhead stays visible.
            gap = 0.08 * (span[1] - span[0])
            unit = direction / length
            ax.annotate(
                "",
                xy=end - unit * gap,
                xytext=start + unit * gap,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#333333",
                    lw=0.8 + 2.5 * margins[a, b] / widest,
                    alpha=0.75,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=2,
            )
            middle = (start + end) / 2
            ax.annotate(
                str(int(margins[a, b])),
                middle,
                fontsize=8,
                color="#333333",
                ha="center",
                va="center",
                zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    condorcet = profile.condorcet_winner()
    for index, position in enumerate(points):
        ax.scatter(
            *position,
            s=520,
            facecolors=PALETTE[index % len(PALETTE)],
            edgecolors="black" if index != condorcet else "crimson",
            linewidths=1.2 if index != condorcet else 3.0,
            zorder=4,
        )
        ax.annotate(
            f"C{index}",
            position,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )

    if title is None:
        if condorcet is not None:
            title = f"Majority graph - Condorcet winner: C{condorcet}"
        elif profile.has_condorcet_cycle():
            title = "Majority graph - no Condorcet winner (cycle)"
        else:
            title = "Majority graph - no Condorcet winner"
    _finish(ax, span, title, legend=False, standalone=standalone)
    return ax


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def plot_winner_distances(distances, bins: int = 30, ax=None, title: str | None = None):
    """Histogram of how far apart the rules' winners stood, across elections."""
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.hist(distances, bins=bins, color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Mean distance between winners")
    ax.set_ylabel("Elections")
    ax.set_title(title or "How far apart the rules put their winners")

    if standalone:
        plt.tight_layout()
        plt.show()
    return ax


def plot_winner_shift(
    candidates, before: dict, after: dict, *, bounds=(-10.0, 10.0), ax=None
):
    """Where each rule's winner sat before, and where it sits now.

    A hollow ring marks the old winner, a filled dot the new one, and an arrow
    joins them when the rule changed its mind.
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(9, 8))

    points = _positions(candidates)
    proxies, labelled = [], set()

    for order, key in enumerate(before):
        colour = PALETTE[order % len(PALETTE)]
        old, new = before[key], after[key]

        if old != new:
            ax.annotate(
                "",
                xy=points[new],
                xytext=points[old],
                arrowprops=dict(arrowstyle="->", color=colour, lw=1.6, alpha=0.8),
                zorder=3,
            )
        ax.scatter(
            *points[old],
            marker="o",
            s=200,
            facecolors="none",
            edgecolors=colour,
            linewidths=2.2,
            zorder=4,
        )
        ax.scatter(
            *points[new],
            marker="o",
            s=170,
            facecolors=colour,
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        for index in (old, new):
            if index not in labelled:
                _label_candidate(ax, index, points[index], bounds)
                labelled.add(index)

        moved = f"C{old} → C{new}" if old != new else f"C{new} (unchanged)"
        proxies.append(_proxy("o", colour, f"{rule_name(key)}: {moved}"))

    proxies.append(_proxy("o", "none", "before"))
    proxies.append(_proxy("o", CANDIDATE_DARK, "after"))
    _finish(ax, bounds, "Winners before and after", proxies, standalone=standalone)
    return ax
