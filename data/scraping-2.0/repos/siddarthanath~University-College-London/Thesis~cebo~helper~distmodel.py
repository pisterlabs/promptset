"""
This file stores distribution models corresponding to predictions from OpenAI.
"""
# -------------------------------------------------------------------------------------------------------------------- #

# Standard Library

# Third Party
import numpy as np
from dataclasses import dataclass

# Private


# -------------------------------------------------------------------------------------------------------------------- #


@dataclass
class DiscreteDist:
    values: np.ndarray
    probs: np.ndarray

    def __post_init__(self):
        # make sure np arrays
        self.values = np.array(self.values)
        self.probs = np.array(self.probs)
        uniq_values = np.unique(self.values)
        if len(uniq_values) < len(self.values):
            # need to mergefg
            uniq_probs = np.zeros(len(uniq_values))
            for i, v in enumerate(uniq_values):
                uniq_probs[i] = np.sum(self.probs[self.values == v])
            self.values = uniq_values
            self.probs = uniq_probs

    def sample(self):
        return np.random.choice(self.values, p=self.probs)

    def mean(self):
        return np.sum(self.values * self.probs)

    def mode(self):
        return self.values[np.argmax(self.probs)]

    def std(self):
        return np.sqrt(np.sum((self.values - self.mean()) ** 2 * self.probs))

    def __repr__(self):
        return f"DiscreteDist({self.values}, {self.probs})"

    def __len__(self):
        return len(self.values)


@dataclass
class GaussDist:
    _mean: float
    _std: float

    def sample(self):
        return np.random.normal(self._mean, self._std)

    def mean(self):
        return self._mean

    def mode(self):
        return self._mean

    def std(self):
        return self._std

    def set_std(self, value):
        self._std = value

    def __repr__(self):
        return f"GaussDist({self._mean}, {self._std})"

    def __len__(self):
        return 1
