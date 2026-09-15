"""Tests for the voting package.

The first group pins the new rules against the three implementations they
replace, computed here exactly as the old ``strategy/`` modules did. The rest
check the things the old code could not express at all: the majority matrix,
the Condorcet methods, STV, and the paradoxes the research notes are about.

Run with ``python tests/test_voting.py`` (no pytest needed).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voting import (  # noqa: E402
    Candidate,
    CopelandRule,
    Election,
    InstantRunoffRule,
    KApprovalRule,
    LpModel,
    MaxminRule,
    PluralityRule,
    Profile,
    RankedPairsRule,
    ScoringRule,
    Voter,
    is_borda_paradox,
    register_rule,
    satisfies_condorcet,
    satisfies_majority,
)
from voting.rules import BordaRule, VetoRule  # noqa: E402

PLURALITY, BORDA, VETO = PluralityRule(), BordaRule(), VetoRule()


def random_election(n_voters, n_candidates, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(-10, 10, (n_voters, 2)),
        rng.uniform(-10, 10, (n_candidates, 2)),
    )


# --------------------------------------------------------------------------
# The replaced implementations still agree
# --------------------------------------------------------------------------


def old_scores(voter_positions, candidate_positions):
    """Plurality, Borda and veto exactly as the old strategy/ modules wrote them."""
    distances = np.linalg.norm(
        voter_positions[:, None, :] - candidate_positions[None, :, :], axis=2
    )
    n_voters, n_candidates = distances.shape

    plurality = np.bincount(np.argmin(distances, axis=1), minlength=n_candidates)
    rank = np.argsort(np.argsort(distances, axis=1), axis=1)
    borda = ((n_candidates - 1) - rank).sum(axis=0)
    vetoes = np.bincount(np.argmax(distances, axis=1), minlength=n_candidates)
    veto = n_voters - vetoes
    return plurality, borda, veto


def test_rules_match_the_old_implementations():
    for n_voters, n_candidates, seed in [(300, 4, 0), (150, 7, 1), (80, 3, 2)]:
        voters, candidates = random_election(n_voters, n_candidates, seed)
        profile = Profile.from_positions(voters, candidates)
        plurality, borda, veto = old_scores(voters, candidates)

        assert np.array_equal(PLURALITY.scores(profile), plurality)
        assert np.array_equal(BORDA.scores(profile), borda)
        assert np.array_equal(VETO.scores(profile), veto)


def test_election_wires_rules_to_positions():
    voters, candidates = random_election(200, 5, 3)
    election = Election(candidates, voters)
    result = election.compare([PLURALITY, BORDA, VETO])

    plurality, borda, veto = old_scores(voters, candidates)
    assert np.array_equal(result.tally("plurality").scores, plurality)
    assert np.array_equal(result.tally(BORDA).scores, borda)
    assert np.array_equal(result.tally("veto").scores, veto)

    assert result.winner_index("borda") == int(np.argmax(borda))
    assert result.winner(BORDA) is result.candidates[int(np.argmax(borda))]
    assert set(result.winners()) == {"plurality", "borda", "veto"}


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


def test_counts_round_trip():
    counts = {(0, 1, 2): 5, (1, 2, 0): 4, (2, 0, 1): 3}
    profile = Profile.from_counts(counts)
    assert profile.n_voters == 12
    assert profile.counts() == counts


def test_ranks_invert_rankings():
    profile = Profile.from_counts({(2, 0, 1): 1, (1, 2, 0): 1})
    # ranks[i, c] is where candidate c sits for voter i.
    assert list(profile.ranks[0]) == [1, 2, 0]
    assert list(profile.ranks[1]) == [2, 0, 1]


def test_pairwise_counts_duels():
    profile = Profile.from_counts({(0, 1, 2): 5, (1, 2, 0): 4, (2, 0, 1): 3})
    assert (profile.pairwise[0, 1], profile.pairwise[1, 0]) == (8, 4)
    assert (profile.pairwise[1, 2], profile.pairwise[2, 1]) == (9, 3)
    assert (profile.pairwise[2, 0], profile.pairwise[0, 2]) == (7, 5)


def test_borda_is_the_row_sum_of_the_majority_matrix():
    voters, candidates = random_election(120, 5, 4)
    profile = Profile.from_positions(voters, candidates)
    assert np.array_equal(BORDA.scores(profile), profile.pairwise.sum(axis=1))


def test_mcgarvey_builds_any_tournament():
    # A three-way cycle: 0 beats 1 beats 2 beats 0.
    wanted = np.array(
        [[False, True, False], [False, False, True], [True, False, False]]
    )
    profile = Profile.from_tournament(wanted)
    assert np.array_equal(profile.beats(), wanted)
    assert profile.condorcet_winner() is None
    assert profile.has_condorcet_cycle()


def test_condorcet_winner_and_loser():
    profile = Profile.from_counts({(0, 1, 2): 6, (1, 0, 2): 3, (2, 1, 0): 2})
    assert profile.condorcet_winner() == 0
    assert profile.condorcet_loser() == 2
    assert not profile.has_condorcet_cycle()


# --------------------------------------------------------------------------
# Rules the old code could not express
# --------------------------------------------------------------------------


def test_condorcet_methods_on_the_classic_cycle():
    profile = Profile.from_counts({(0, 1, 2): 5, (1, 2, 0): 4, (2, 0, 1): 3})
    assert profile.condorcet_winner() is None

    # Everyone wins one duel and loses one, so Copeland cannot separate them.
    assert list(CopelandRule().scores(profile)) == [0, 0, 0]
    # Worst support is 5, 4, 3 - candidate 0 has the least bad defeat.
    assert list(MaxminRule().scores(profile)) == [5, 4, 3]
    # Margins are 1>2 by 6, 0>1 by 4, 2>0 by 2; the last would cycle and is skipped.
    assert list(RankedPairsRule().scores(profile)) == [2, 1, 0]


def test_instant_runoff_squeezes_out_the_plurality_leader():
    profile = Profile.from_counts({(0, 1, 2): 4, (1, 0, 2): 3, (2, 1, 0): 5})
    assert PLURALITY.winner(profile) == 2, "plurality crowns 2"

    # 1 has the fewest first places and goes first; its voters transfer to 0,
    # which then leads 7 to 5.
    scores = InstantRunoffRule().scores(profile)
    assert int(np.argmax(scores)) == 0
    assert int(np.argmin(scores)) == 1


def test_k_approval_spans_plurality_to_veto():
    voters, candidates = random_election(200, 5, 5)
    profile = Profile.from_positions(voters, candidates)
    assert np.array_equal(KApprovalRule(1).scores(profile), PLURALITY.scores(profile))
    assert np.array_equal(KApprovalRule(4).scores(profile), VETO.scores(profile))


def test_social_ranking_orders_every_candidate():
    profile = Profile.from_counts({(0, 1, 2): 5, (1, 2, 0): 4, (2, 0, 1): 3})
    assert MaxminRule().social_ranking(profile) == [0, 1, 2]


# --------------------------------------------------------------------------
# Criteria from the research notes
# --------------------------------------------------------------------------


def test_criteria_report_what_the_notes_claim():
    # Three voters rank 0 first and two rank 1 first, so 0 holds an outright
    # majority and beats both others head to head. Borda still prefers 1:
    #   Borda(0) = 3*2 = 6,  Borda(1) = 3*1 + 2*2 = 7,  Borda(2) = 2*1 = 2.
    profile = Profile.from_counts({(0, 1, 2): 3, (1, 2, 0): 2})
    assert list(BORDA.scores(profile)) == [6, 7, 2]
    assert profile.condorcet_winner() == 0

    assert satisfies_majority(PLURALITY, profile) is True
    assert satisfies_majority(BORDA, profile) is False, "Borda fails the majority criterion"
    assert satisfies_condorcet(PLURALITY, profile) is True
    assert satisfies_condorcet(BORDA, profile) is False, "and the Condorcet criterion"

    cycle = Profile.from_counts({(0, 1, 2): 5, (1, 2, 0): 4, (2, 0, 1): 3})
    assert satisfies_condorcet(BORDA, cycle) is None, "no Condorcet winner to elect"


def test_borda_paradox_is_detected():
    # 0 leads on first preferences but loses both duels: the Borda paradox.
    profile = Profile.from_counts({(0, 1, 2): 4, (1, 2, 0): 3, (2, 1, 0): 3})
    assert PLURALITY.winner(profile) == 0
    assert profile.condorcet_loser() == 0
    assert is_borda_paradox(profile)
    assert not is_borda_paradox(Profile.from_counts({(0, 1, 2): 5, (0, 2, 1): 4}))


# --------------------------------------------------------------------------
# Extending the library
# --------------------------------------------------------------------------


class TopTwoRule(ScoringRule):
    """Invented here: 3 points for first place, 1 for second, 0 after."""

    key, name = "top_two", "Top two"

    def weights(self, n_candidates: int) -> np.ndarray:
        weights = np.zeros(n_candidates, dtype=int)
        weights[:2] = [3, 1][: min(2, n_candidates)]
        return weights


def test_a_new_rule_is_one_class():
    register_rule(TopTwoRule(), replace=True)
    rule = TopTwoRule()
    profile = Profile.from_counts({(0, 1, 2): 3, (1, 2, 0): 2, (2, 1, 0): 2})

    # 0: 3*3 = 9;  1: 3*1 + 2*3 + 2*1 = 11;  2: 2*1 + 2*3 = 8
    assert list(rule.scores(profile)) == [9, 11, 8]
    assert rule.winner(profile) == 1

    voters, candidates = random_election(100, 4, 6)
    result = Election(candidates, voters).compare([rule, PLURALITY])
    assert set(result.winners()) == {"top_two", "plurality"}


# --------------------------------------------------------------------------
# The linear program
# --------------------------------------------------------------------------


def feasible_targets(n_candidates, n_voters, seed):
    """An election with no ties, so its own winners are reachable by the LP."""
    rng = np.random.default_rng(seed)
    while True:
        candidates = rng.uniform(-9, 9, (n_candidates, 2))
        voters = rng.uniform(-10, 10, (n_voters, 2))
        result = Election(candidates, voters).compare([PLURALITY, BORDA])
        if not result.has_tie():
            return candidates, voters, result.winner_indices()


def test_lp_builds_an_electorate_matching_its_targets():
    candidates, _, winners = feasible_targets(4, 120, 10)
    model = LpModel(
        candidates=[Candidate(id=i, position=p) for i, p in enumerate(candidates)],
        winners=winners,
        n_voters=120,
        pool_size=60_000,
        rng=np.random.default_rng(11),
    )
    assert model.solve() == "Optimal", model.status
    assert model.check() == winners, "the solved profile misses the targets"

    positions = model.generate_voter_positions()
    assert len(positions) == 120
    realized = Election(candidates, positions).compare([PLURALITY, BORDA])
    assert realized.winner_indices() == winners, "generated voters miss the targets"


def test_lp_can_add_voters_to_an_existing_electorate():
    candidates, voters, winners = feasible_targets(4, 80, 20)
    target = next(j for j in range(4) if j != winners["plurality"])

    model = LpModel(
        candidates=candidates,
        winners={"plurality": target},
        base_voters=[Voter(position=p) for p in voters],
        max_added=400,
        objective="min_total",
        pool_size=60_000,
        rng=np.random.default_rng(21),
    )
    assert model.solve() == "Optimal", model.status
    assert model.check()["plurality"] == target

    added = model.generate_voter_positions()
    combined = np.vstack([voters, added]) if len(added) else voters
    assert Election(candidates, combined).run(PLURALITY).winner_index == target


def test_lp_can_demand_a_condorcet_winner():
    rng = np.random.default_rng(30)
    candidates = rng.uniform(-9, 9, (4, 2))
    for target in range(4):
        model = LpModel(
            candidates=candidates,
            condorcet_winner=target,
            n_voters=100,
            pool_size=60_000,
            rng=np.random.default_rng(31),
        )
        if model.solve() != "Optimal":
            continue
        assert model.solution_profile().condorcet_winner() == target
        return
    raise AssertionError("no candidate could be made a Condorcet winner")


def test_lp_rejects_impossible_targets():
    candidates = np.random.default_rng(40).uniform(-9, 9, (4, 2))
    for winners, expected in [
        ({"borda": 9}, ValueError),  # out of range
        ({"borda": -1}, ValueError),  # would quietly report Infeasible
        ({"bordaa": 1}, KeyError),  # typo
        ({"copeland": 1}, TypeError),  # not a scoring rule
    ]:
        try:
            LpModel(candidates=candidates, winners=winners, n_voters=50)
        except expected:
            continue
        raise AssertionError(f"{winners} should have raised {expected.__name__}")


# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = []
    for name, test in tests:
        try:
            test()
        except Exception as error:  # noqa: BLE001 - a runner reports everything
            failures.append(name)
            print(f"FAIL  {name}\n      {type(error).__name__}: {error}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
