from dataclasses import dataclass

import numpy as np


# eq=False: identity equality, so candidates.index(winner) keeps working and we
# never compare ndarray positions with == (which would be ambiguous).
@dataclass(eq=False)
class Candidate:
    id: int
    position: np.ndarray
