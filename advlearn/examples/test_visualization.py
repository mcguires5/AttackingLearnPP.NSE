from advlearn.utils import PoisonViz
from advlearn.attacks.poison import SVMAttack, LogisticAttack, LinearAttack
from sklearn.datasets.samples_generator import make_blobs
import numpy as np

random_state = 0

# Setup Boundary Regions
mesh = 5
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

# Generate Attack Params

attack_class = LinearAttack
args = []
kwargs = {'boundary': boundary, 'lasso_lambda': 0.25, 'step_size': 0.1, 'max_steps': 100}

#attack_class = SVMAttack
#args = []
#kwargs = {'boundary': boundary, 'step_size': 0.5, 'max_steps': 100,
#          'c': 1, 'kernel': 'rbf', 'degree': 3, 'coef0': 1, 'gamma': 1}

#attack_class = LogisticAttack
#args = []
#kwargs = {'opt_method': 'GD', 'boundary': boundary, 'step_size': 0.1, 'max_steps': 100}

# Generate and Plot Visualization
viz = PoisonViz(attack_class, args, kwargs, X, y, X, y)
viz.plot_loss([1, 2, 1], mesh_x_steps, mesh_y_steps, target=1, decision=False)
viz.plot_loss([1, 2, 2], mesh_x_steps, mesh_y_steps, target=-1, decision=False)
viz.show()

print('Done!')
