"""Pictures of elections.

The drawings are the ones the project already had - gray voter dots, dark
candidate dots, colored winner dots, and the colored variant with "x" candidates
and star winners. What changed is that they come from **one** function,
:func:`plot_election`, instead of three near-identical ones that each repeated
the axes setup, the legend, the palette and the winner-marker nesting.

Two additions, both opt-in and off by default:

``regions=True``          outlines the plurality regions in thin black lines, so
                          you can see where a plurality win comes from.
:func:`plot_majority_graph`  draws the tournament - an arrow from the winner of
                          each duel to its loser - which is the only way to look
                          at a Condorcet cycle.
"""

import random

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.spatial import Voronoi

from voting.profile import Profile
from voting.rules import rule_name

PALETTE = plt.cm.tab10.colors

# Every picture leaves this much room around `bounds`, so candidates sitting on
# the edge are not cut in half. Region borders are drawn out to the frame, so
# they need the same number.
FRAME_MARGIN = 0.5


# --------------------------------------------------------------------------
# Random points
# --------------------------------------------------------------------------


def random_2d_point(bounds=(-10, 10)) -> np.ndarray:
    """One uniform point in the square ``bounds``."""
    return np.array([random.uniform(*bounds), random.uniform(*bounds)])


def random_2d_points(n: int = 5, bounds=(-10, 10)) -> np.ndarray:
    """``(n, 2)`` uniform points in the square ``bounds``."""
    return np.array([random_2d_point(bounds) for _ in range(n)])


def plot(candidates: np.ndarray, voters: np.ndarray):
    """The bare scatter: green crosses for candidates, dots for voters."""
    plt.scatter(
        candidates[:, 0],
        candidates[:, 1],
        color="green",
        marker="x",
        label="Candidates",
    )
    plt.scatter(voters[:, 0], voters[:, 1], marker="o", label="Voters")


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


def _proxy(marker: str, facecolor, label: str):
    """A legend entry for something drawn with scatter."""
    return mlines.Line2D(
        [],
        [],
        marker=marker,
        linestyle="None",
        markerfacecolor=facecolor,
        markeredgecolor="black",
        markersize=11,
        markeredgewidth=1.2,
        label=label,
    )


def _label_candidate(ax, index: int, position, bounds) -> None:
    """Name a candidate, below the marker when it would overflow the top edge."""
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


def _strip_axes(ax) -> None:
    """Drop the tick marks and numbers, keep the frame."""
    ax.tick_params(
        axis="both", which="both", length=0, labelbottom=False, labelleft=False
    )


def _finish(ax, bounds, title, extra_handles=(), legend=True, standalone=False):
    """The frame and legend every election picture shares."""
    ax.set_xlim(bounds[0] - FRAME_MARGIN, bounds[1] + FRAME_MARGIN)
    ax.set_ylim(bounds[0] - FRAME_MARGIN, bounds[1] + FRAME_MARGIN)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    _strip_axes(ax)
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


def _clip_segment(start, end, bounds):
    """The part of a segment inside the square ``bounds``, or None if it misses.

    Liang-Barsky: walk the parameter ``t`` along the segment and keep shrinking
    the admissible interval with each of the four edges.
    """
    lo, hi = bounds
    start = np.asarray(start, dtype=float)
    delta = np.asarray(end, dtype=float) - start
    t_enter, t_leave = 0.0, 1.0

    for axis in (0, 1):
        if abs(delta[axis]) < 1e-12:  # parallel to this pair of edges
            if not lo <= start[axis] <= hi:
                return None
            continue
        first = (lo - start[axis]) / delta[axis]
        second = (hi - start[axis]) / delta[axis]
        t_enter = max(t_enter, min(first, second))
        t_leave = min(t_leave, max(first, second))

    if t_enter > t_leave:
        return None
    return start + t_enter * delta, start + t_leave * delta


def _region_borders(candidates: np.ndarray, bounds) -> list:
    """The exact Voronoi boundaries as straight segments clipped to ``bounds``.

    Every boundary lies on the perpendicular bisector of two candidates, so the
    segments are exact - no grid, no staircase. ``scipy.spatial.Voronoi`` gives
    the finite ones directly; a ridge that runs off to infinity is replaced by a
    segment long enough to leave the box, pointing away from the centre of the
    candidate cloud, and then clipped like the rest.
    """
    frame = (bounds[0] - FRAME_MARGIN, bounds[1] + FRAME_MARGIN)
    span = frame[1] - frame[0]

    if len(candidates) < 2:
        return []
    if len(candidates) == 2:
        # Qhull needs three points; with two the single boundary is the whole
        # perpendicular bisector.
        tangent = candidates[1] - candidates[0]
        normal = np.array([-tangent[1], tangent[0]])
        normal = normal / np.linalg.norm(normal)
        middle = candidates.mean(axis=0)
        raw = [(middle - normal * 10 * span, middle + normal * 10 * span)]
    else:
        try:
            diagram = Voronoi(candidates)
        except Exception:  # collinear or duplicated candidates
            return []
        centre = candidates.mean(axis=0)
        raw = []
        for (left, right), ends in zip(
            diagram.ridge_points, diagram.ridge_vertices
        ):
            first, second = ends
            if first >= 0 and second >= 0:
                raw.append((diagram.vertices[first], diagram.vertices[second]))
                continue
            finite = diagram.vertices[first if first >= 0 else second]
            tangent = candidates[right] - candidates[left]
            tangent = tangent / np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = candidates[[left, right]].mean(axis=0)
            outward = np.sign(np.dot(midpoint - centre, normal)) * normal
            raw.append((finite, finite + outward * 10 * span))

    clipped = [_clip_segment(a, b, frame) for a, b in raw]
    return [segment for segment in clipped if segment is not None]


def _draw_region_borders(ax, candidates: np.ndarray, bounds) -> None:
    """Outline the plurality regions with thin straight black lines."""
    for start, end in _region_borders(candidates, bounds):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="black",
            linewidth=0.8,
            solid_capstyle="projecting",
            zorder=1,
        )


# --------------------------------------------------------------------------
# The election picture
# --------------------------------------------------------------------------


def plot_election(
    candidates,
    voters=None,
    winners: dict | None = None,
    *,
    added_voters=None,
    color_voters: bool = False,
    regions: bool = False,
    bounds: tuple[float, float] = (-10.0, 10.0),
    title: str | None = None,
    ax=None,
    legend: bool = True,
):
    """Draw candidates, voters and - if given - who won under which rule.

    candidates    Candidate objects or an ``(M, 2)`` array.
    voters        Voter objects or an ``(N, 2)`` array; may be omitted.
    winners       ``{rule key: candidate index}``, e.g. ``result.winner_indices()``.
    added_voters  A second group drawn in green on top, for the LP models that
                  add voters to an existing electorate.
    color_voters  Color each voter by the candidate they rank first; candidates
                  then become colored "x" and winners become stars.
    regions       Outline the plurality regions in thin black lines. Off by
                  default.
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
    extending = added_voters is not None

    if regions and len(candidate_positions):
        _draw_region_borders(ax, candidate_positions, bounds)

    # --- voters ---
    if color_voters and len(voter_positions):
        choices = Profile.from_positions(
            voter_positions, candidate_positions
        ).top_choices()
        for index in range(len(candidate_positions)):
            group = voter_positions[choices == index]
            if not len(group):
                continue
            ax.scatter(
                group[:, 0],
                group[:, 1],
                color=PALETTE[index % len(PALETTE)],
                s=60,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.4,
                label=f"Top choice: C{index} ({len(group)} voters)",
            )
    elif len(voter_positions):
        ax.scatter(
            voter_positions[:, 0],
            voter_positions[:, 1],
            marker="o",
            color="gray",
            s=45,
            alpha=0.4,
            label=(
                f"Existing voters ({len(voter_positions)})"
                if extending
                else f"Voters ({len(voter_positions)})"
            ),
        )

    if len(added_positions):
        ax.scatter(
            added_positions[:, 0],
            added_positions[:, 1],
            color="green",
            s=55,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
            label=f"Added voters ({len(added_positions)})",
        )

    # --- candidates ---
    # In colored mode a non-winner is a colored "x"; a winner's marker becomes a
    # star (drawn below) so the symbols never stack. In plain mode every
    # candidate is a dot, dark for non-winners and colored for winners.
    winning = set(winners.values())
    labelled = False
    for index, position in enumerate(candidate_positions):
        if index not in winning:
            if color_voters:
                ax.scatter(
                    *position,
                    marker="x",
                    s=150,
                    c=[PALETTE[index % len(PALETTE)]],
                    linewidths=3,
                    zorder=4,
                )
            else:
                ax.scatter(
                    *position,
                    marker="o",
                    s=90,
                    c="#555555",
                    zorder=4,
                    label=None if labelled else "Candidate",
                )
                labelled = True
        _label_candidate(ax, index, position, bounds)

    # --- winners ---
    # Colored mode: a star in the candidate's own color, so it stands out inside
    # a same-colored cluster. Plain mode: a dot in a per-rule color. Either way
    # the marks nest (smaller on top) when rules share a candidate.
    proxies, seen = [], {}
    for order, (key, index) in enumerate(winners.items()):
        level = seen.get(index, 0)
        seen[index] = level + 1
        if color_voters:
            marker = "*"
            colour = PALETTE[index % len(PALETTE)]
            size = 320 + level * 320
        else:
            marker = "o"
            colour = PALETTE[order % len(PALETTE)]
            size = 160 + level * 180
        ax.scatter(
            *candidate_positions[index],
            marker=marker,
            s=size,
            facecolors=colour,
            edgecolors="black",
            linewidths=1.2,
            zorder=5 - level * 0.1,
        )
        proxies.append(
            _proxy(marker, colour, f"Winner – {rule_name(key)} (C{index})")
        )

    if title is None:
        total = len(voter_positions) + len(added_positions)
        counts = (
            f"{total} voters: {len(voter_positions)} + {len(added_positions)} added"
            if extending
            else f"{total} voters"
        )
        title = f"Voter Distribution in Euclidean Space ({counts})"
    _finish(ax, bounds, title, proxies, legend=legend, standalone=standalone)
    return ax


def plot_result(result, **kwargs):
    """Draw an :class:`~voting.election.ElectionResult`.

    A shortcut for :func:`plot_election` that unpacks the result's candidates,
    voters and winners.
    """
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
    circle otherwise. Arrow width grows with the margin, and a Condorcet winner -
    all arrows outgoing - is ringed in red. When there is none, the cycle is
    visible as a loop.
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
    gap = 0.08 * (span[1] - span[0])

    for a in range(n):
        for b in range(n):
            if margins[a, b] <= 0:
                continue
            start, end = points[a], points[b]
            direction = end - start
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            unit = direction / length  # stop short so the arrowhead stays visible
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
            ax.annotate(
                str(int(margins[a, b])),
                (start + end) / 2,
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


def plot_winner_distances(distances, bins: int = 5, ax=None):
    """Histogram of how far apart the rules' winners stood, across elections."""
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
    return ax
