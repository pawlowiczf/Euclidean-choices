from election.result import ElectionResult


class ResultsAnalyzer:
    def __init__(self, results: list[ElectionResult]):
        self.results = results
