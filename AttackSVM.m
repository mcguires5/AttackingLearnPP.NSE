clear classes

random_state = 0;
% Setup Boundary Regions
mesh = 10.5
step = 0.25;
bound = mesh-0.5;
boundary = [[-bound -bound];[bound bound]];
mesh_x_steps = -mesh:mesh:step;
mesh_y_steps = -mesh:mesh:step;

% Generate Dataset
centers = [[-1. -0.7]; [1. 0.7]];
%commandStr = 'python C:/Users/baseb_000/AppData/Local/Programs/Python/Python36/Lib/site-packages/sklearn/datasets/samples_generator.py';
% [status, commandOut] = system(commandStr);
samples_generator = py.importlib.import_module('sklearn.datasets.samples_generator'); 
AllArgs = pyargs('n_samples',int32(50),'centers',centers,'cluster_std',1,'n_features',2,'random_state',int32(random_state));
Output = samples_generator.make_blobs(AllArgs);
X = double(Output{1});
y = double(Output{2});
y = (y * 2) - 1;

X = vertcat(X, [-4 0; -2 -4; -2 4; 4 0]);
y = vertcat(y', [-1; -1; -1; 1]);
y = int32(y);
% Generate Attack Params

% attack_class = LinearAttack
% args = []
% kwargs = {'boundary': boundary, 'lasso_lambda': 0.25, 'step_size': 0.1, 'max_steps': 100}

np = py.importlib.import_module('numpy'); 

boundary = np.array(boundary);
target = 1;
q = 1;
p = py.sys.path;
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\')
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\advlearn\attacks\poison\')
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\advlearn\attacks\')
Poison = py.importlib.import_module('advlearn.attacks.poison'); 
py.importlib.reload(Poison);
%attack_class = Poison.SVMAttack;
kwargs = pyargs('boundary', boundary, 'step_size', 0.5, 'max_steps', int32(100),'c', int32(1), 'kernel', 'rbf', 'degree', 3, 'coef0', 1, 'gamma', 1);
attack = py.advlearn.attacks.poison.SVMAttack(kwargs);
args = pyargs('data', np.array(X), 'labels', np.array(y));
Poison.SVMAttack.fit(attack,args);
kwargs = pyargs('self',attack,'n_points', int32(3));
temp = Poison.SVMAttack.attack(kwargs);
% x_attack, y_attack = model.get_attack_point()
AttackPoints = double(temp{1});

disp(AttackPoints)
disp(temp(2))
hold on
plot(X(y == -1,1),X(y == -1,2),'r*')
plot(X(y == 1,1),X(y == 1,2),'b*')
plot(AttackPoints(:,1), AttackPoints(:,2), 'b+', 'MarkerSize', 20)
hold off
disp('Done!')