"""Run general linear attack"""

import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from advlearn.attacks.poison import LogisticAttack


random_state = 0

# Setup Boundary Regions
mesh = 7
step = 0.25
bound = mesh-0.5
boundary = np.array([[-bound, -bound],
                     [bound, bound]])
mesh_x_steps = np.arange(-mesh, mesh, step)
mesh_y_steps = np.arange(-mesh, mesh, step)

# Generate Dataset
centers = [[-1, -0.7], [1, 0.7]]
X, y = make_blobs(n_samples=50, centers=centers, cluster_std=1,
                  n_features=2, random_state=random_state)
y = (y * 2) - 1

X = np.append(X, np.array([[-4, 0], [-2, -4], [-2, 4], [4, 0]]), axis=0)
y = np.append(y, np.array([-1, -1, -1, 1]))

x_c = np.array([0, 0])
y_c = np.array([-1])

# HJKFDASF
model = LogisticAttack()
model.fit(X, y)
x_attack, y_attack = model.attack(1)
trajectory = model.attack_trajectory(x_c, y_c)

print(x_attack)
print(y_attack)
