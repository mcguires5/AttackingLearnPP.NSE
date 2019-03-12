% @Author: delengowski
% @Date:   2019-03-06 17:52:35
% @Last Modified by:   delengowski
% @Last Modified time: 2019-03-06 18:06:41
function [attackPoints,attackLabels] = chrisAttacks(data,labels,boundary,kwargs,numberAttackPoints)
	% get input data
	y = int32(labels);
	x = data;
    np = py.importlib.import_module('numpy');
	boundary = np.array(boundary);
	Poison = py.importlib.import_module('advlearn.attacks.poison'); 
	py.importlib.reload(Poison);
	% Set up attack
	attack = py.advlearn.attacks.poison.SVMAttack(kwargs);
	args = pyargs('data', np.array(x), 'labels', np.array(y));
	% Fit the data??
	Poison.SVMAttack.fit(attack,args);
	kwargs = pyargs('self',attack,'n_points', int32(3));
	% Get attack data
	attackData = Poison.SVMAttack.attack(kwargs);
	attackPoints = double(attackData{1});
	attackLabels = double(attackLabels{2});
end