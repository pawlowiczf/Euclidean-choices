"""Voting rules over Euclidean elections.

Voters and candidates are points in the plane. Each voter ranks the candidates
by distance, and from there on the rules are ordinary social choice functions:
they read a :class:`~voting.profile.Profile` of rankings and know nothing about
geometry. That split is what lets Condorcet methods and STV live here alongside
plurality and Borda.

    from voting import Election, Candidate, Voter, PluralityRule, BordaRule
    from voting.plotting import plot_result

    election = Election(candidates, voters)
    result = election.compare([PluralityRule(), BordaRule()])
    result.winners()          # {rule key: Candidate}
    result.winner_indices()   # {rule key: index}  - also the LP's target format
    plot_result(result, regions=True)

Modules
-------
``profile``   Profile: the rankings, and the majority matrix rules read.
``rules``     the rules, their three families, and the registry.
``election``  Candidate, Voter, Election, Tally, ElectionResult.
``lp``        integer programs that build an electorate electing chosen winners.
``analysis``  winner distances, and the classical criteria.
``plotting``  pictures, including plurality regions and the majority graph.

Adding a rule is one subclass - see :mod:`voting.rules`.
"""

from voting.analysis import (
    ResultsAnalyzer,
    find_farthest_pair,
    find_farthest_triple,
    is_borda_paradox,
    mean_pairwise_distance,
    satisfies_condorcet,
    satisfies_majority,
)
from voting.election import Candidate, Election, ElectionResult, Tally, Voter
from voting.lp import (
    OBJECTIVES,
    LpModel,
    exclude_current_solution,
    exclude_current_solution_bigm,
    exclude_largest_variable,
    sample_regions,
)
from voting.profile import Profile
from voting.rules import (
    CLASSIC_RULES,
    RULES,
    BordaRule,
    CondorcetRule,
    CopelandRule,
    CustomScoringRule,
    DowdallRule,
    InstantRunoffRule,
    KApprovalRule,
    MaxminRule,
    PluralityRule,
    RankedPairsRule,
    Rule,
    ScoringRule,
    VetoRule,
    register_rule,
    rule_for_key,
    rule_key,
    rule_name,
)

__all__ = [
    # elections
    "Election",
    "ElectionResult",
    "Candidate",
    "Voter",
    "Tally",
    "Profile",
    # rules
    "Rule",
    "ScoringRule",
    "CondorcetRule",
    "PluralityRule",
    "BordaRule",
    "VetoRule",
    "KApprovalRule",
    "DowdallRule",
    "CustomScoringRule",
    "CopelandRule",
    "MaxminRule",
    "RankedPairsRule",
    "InstantRunoffRule",
    "CLASSIC_RULES",
    "RULES",
    "register_rule",
    "rule_for_key",
    "rule_key",
    "rule_name",
    # building electorates
    "LpModel",
    "OBJECTIVES",
    "sample_regions",
    "exclude_current_solution",
    "exclude_largest_variable",
    "exclude_current_solution_bigm",
    # measurements and criteria
    "ResultsAnalyzer",
    "mean_pairwise_distance",
    "find_farthest_pair",
    "find_farthest_triple",
    "satisfies_condorcet",
    "satisfies_majority",
    "is_borda_paradox",
]
