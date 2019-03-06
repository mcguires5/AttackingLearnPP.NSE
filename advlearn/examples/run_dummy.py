"""Run Dummy

This is an example of how to use this advlearn using all dummy functions.
"""

from advlearn.defenses.dummy import DummyDefense
import numpy as np


X = np.array([[-0.12840393, 0.66446571], [1.32319756, -0.13181616],
              [0.04296502, -0.37981873], [0.83631853, 0.18569783]])
Y = np.array([1, 2, 2, 2])

defense = DummyDefense()
out, used = defense.transform(X)

print(defense)
print(out)
print(used)
