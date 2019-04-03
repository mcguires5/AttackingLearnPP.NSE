% @Author: delengowski
% @Date:   2019-03-06 17:52:35
% @Last Modified by:   Delengowski_Mobile
% @Last Modified time: 2019-04-02 13:54:53
function [attackPoints,attackLabels] = chrisAttacks(data,labels,atkSet)
	% get input data
    labels(labels == 2) = -1; % Convert labels of class 2 to -1 for svm
	y = int32(labels);
	x = data;
    np = py.importlib.import_module('numpy');
	boundary = np.array(atkSet.boundary);
	Poison = py.importlib.import_module('advlearn.attacks.poison'); 
	%'boundary', boundary,
	svmPoisonAttackArgs = pyargs(...
							'step_size', atkSet.step_size,...
							 'max_steps', int32(atkSet.max_steps),...
							 'c', int32(atkSet.c),...
							 'kernel', atkSet.kernel,...
							 'degree', atkSet.degree,...
							  'coef0', atkSet.coef0,...
							  'gamma', atkSet.gamma);
	py.importlib.reload(Poison);
	% Set up attack
	attack = py.advlearn.attacks.poison.SVMAttack(svmPoisonAttackArgs);
	args = pyargs('data', np.array(x), 'labels', np.array(y));
	% Fit the data??
	Poison.SVMAttack.fit(attack,args);
	kwargs = pyargs('self',attack,'n_points', int32(atkSet.numAtkPts));
	% Get attack data
	attackData = Poison.SVMAttack.attack(kwargs);
	attackPoints = double(attackData{1});
	attackLabels = double(attackData{2});
    attackLabels(attackLabels == -1) = 2; % Convert the -1 labels back to 2 
end