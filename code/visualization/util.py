import matplotlib.pyplot as plt
import numpy as np
import random


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


def plot_results(candidates: list, voters: list, results: dict[str, "Candidate"]):
    candidates_arr = np.array([c.position for c in candidates])
    voters_arr = np.array([v.position for v in voters])

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
    for strategy_name, winner in results.items():
        winner_to_strategies.setdefault(winner.id, (winner, []))[1].append(
            str(strategy_name)
        )

    colors = plt.cm.tab10.colors
    for i, (winner, strategy_names) in enumerate(winner_to_strategies.values()):
        label = "Winner:\n" + "\n".join(strategy_names)
        plt.scatter(*winner.position, marker="*", s=300, color=colors[i], label=label)

    plt.gcf().set_size_inches(8, 7)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
    plt.tight_layout()
    plt.show()
