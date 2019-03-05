random_state = 0;
% Setup Boundary Regions
mesh = 5;
step = 0.25;
bound = mesh-0.5;
boundary = [[-bound -bound],[bound bound]];
mesh_x_steps = -mesh:mesh:step;
mesh_y_steps = -mesh:mesh:step;

% Generate Dataset
centers = [[-1. -0.7]; [1. 0.7]];

samples_generator = py.importlib.import_module('sklearn.datasets.samples_generator'); 
AllArgs = pyargs('n_samples',int32(50),'centers',centers,'cluster_std',1,'n_features',2,'random_state',int32(random_state));
Output = samples_generator.make_blobs(AllArgs);
X = double(Output{1});
y = double(Output{2});
y = (y * 2) - 1;

X = vertcat(X, [-4 0; -2 -4; -2 4; 4 0]);
y = vertcat(y', [-1; -1; -1; 1]);

target = 1;
q = 1;
p = py.sys.path;
insert(p,int32(0),'C:\Users\baseb_000\PycharmProjects\advlearn\attacks\poison\')
Poison = py.importlib.import_module('advlearn.attacks.poison'); 
py.importlib.reload(Poison);
%attack_class = Poison.SVMAttack;
args = [];
%attack = attacks.poison.SVMWrapper(X,y,3);
AllArgs = pyargs('X',X,'y',y);
Poison.SVMAttack.Attack(AllArgs)
print(x_attack)
print(y_attack)
print(x_attack)
print(y_attack)
