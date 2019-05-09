% @Author: delengowski
% @Date:   2019-03-06 17:52:35
% @Last Modified by:   Delengowski_Mobile
% @Last Modified time: 2019-04-02 13:54:53
function [attackPoints,attackLabels] = chrisAttacks(data,labels,atkSet)
    global np;
    global Poison;

	% get input data
    labels(labels == 2) = -1; % Convert labels of class 2 to -1 for svm
    atkSet.ClassToAttack(atkSet.ClassToAttack == 2) = -1;
	y = int32(labels);
	x = data;

	boundary = np.array(atkSet.Boundary);
	
	%'boundary', boundary,
	svmPoisonAttackArgs = pyargs('boundary', boundary,...
							'step_size', atkSet.StepSize,...
							 'max_steps', int32(atkSet.MaxSteps),...
							 'kernel', atkSet.Kernel,...
							 'degree', atkSet.Degree,...
							  'coef0', atkSet.Coef0,...
							  'gamma', atkSet.Gamma);
	%py.importlib.reload(Poison);
	% Set up attack
	attack = py.advlearn.attacks.poison.SVMAttack(svmPoisonAttackArgs);

	args = pyargs('data', np.array(x), 'labels', np.array(y));
	% Fit the data??
	Poison.SVMAttack.fit(attack,args);
	kwargs = pyargs('self',attack,'n_points', int32(atkSet.NumAtkPts),'target_class',int32(atkSet.ClassToAttack));
	% Get attack data
	attackData = Poison.SVMAttack.attack(kwargs);
	attackPoints = double(attackData{1});
	attackLabels = double(attackData{2});
    attackLabels(attackLabels == -1) = 2; % Convert the -1 labels back to 2 
end