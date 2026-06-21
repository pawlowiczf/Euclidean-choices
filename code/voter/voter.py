from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class Voter:
    position: np.ndarray
