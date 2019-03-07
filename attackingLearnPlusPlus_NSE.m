% Generate data or load data??
T = 200;  % number of time stamps
N = 100;  % number of data points at each time
[data_train, labels_train,data_test,labels_test] = ConceptDriftData('checkerboard', T, N);
for t = 1:T
  % i wrote the code along time ago and i used at assume column vectors for
  % data and i wrote all the code for learn++ on github to assume row
  % vectors. the primary reasoning for this is that the stats toolbox in
  % matlab uses row vectors for operations like mean, cov and the
  % classifiers like CART and NB
  data_train{t} = data_train{t}';
  labels_train{t} = labels_train{t}';
  data_test{t} = data_test{t}';
  labels_test{t} = labels_test{t}';
end

% Classifier parameters
model.type = 'SVM';          % base classifier
net.a = .5;                   % slope parameter to a sigmoid
net.b = 10;                   % cutoff parameter to a sigmoid
net.threshold = 0.01;         % how small is too small for error
net.mclass = 2;               % number of classes in the prediciton problem
net.base_classifier = model;  % set the base classifier in the net struct




% These cell arrays are structured as {class}{timeStep}
numClasses = numel(unique(labels_train{1}));
classLabelsIdxs = cell(1,numClasses);
classData = cell(1,numClasses);
classMeans = cell(1,numClasses);
classVariances = cell(1,numClasses);
for iClass = 1:numClasses
	classLabelsIdxs{iClass} = cellfun(@(x) eq(x,1),labels_train,'UniformOutput',false);
	classData{iClass} = cellfun(@(a,b) a(b,:),data_train,classLabelsIdxs{iClass},'UniformOutput',false);
	classMeans{iClass} = cellfun(@(x) mean(x),classData{iClass},'UniformOutput',false);
	classVariances{iClass} = cellfun(@(x) cov(x),classData{iClass},'UniformOutput',false);
end

% Create a SINDy model and means, 1 by number classes SINDy object array
numDims = size(data_train{1},2);
sindyMeans(1,numClasses) = SINDy(); % Object array of sindy objects
[sindyMeans.lambda] = deal(2e-6);
[sindyMeans.polyOrder] = deal(5);
[sindyMeans.useSine] = deal(0);
[sindyMeans.sineMultiplier] = deal(10);
[sindyMeans.useExp] = deal(0);
[sindyMeans.expMultiplier] = deal(10);
[sindyMeans.useCustomPoolData] = deal(1);
[sindyMeans.nonDynamical] = deal(0);
% TVRegDiff parameters for sindy
[sindyMeans.useTVRegDiff] = deal(0);
[sindyMeans.iter] = deal(sindy.iter;) 
[sindyMeans.alph] = deal(sindy.alph);
[sindyMeans.ep] = deal(sindy.ep);
[sindyMeans.scale] = deal(sindy.scale);
[sindyMeans.plotflag] = deal(0);
[sindyMeans.diagflag] = deal(0);
% Repeat for covariances this is accessed (row,column) = size of covariance matrix, and then the depth is classes
% So for 2 dim data, covariance matrix is 2x2, and if there's two classes then sindyCovariances(1,1,1) accesses
% the sindy object for class 1, row 1 column 1 of its covariance matrix.
[heightCovMat widthCovMat] = size(classVariances{1}{1});
sindyCovariances(heightCovMat,widthCovMat,numClasses) = SINDy();
[sindyCovariances.lambda] = deal(2e-6);
[sindyCovariances.polyOrder] = deal(5);
[sindyCovariances.useSine] = deal(0);
[sindyCovariances.sineMultiplier] = deal(10);
[sindyCovariances.useExp] = deal(0);
[sindyCovariances.expMultiplier] = deal(10);
[sindyCovariances.useCustomPoolData] = deal(1);
[sindyCovariances.nonDynamical] = deal(0);
% TVRegDiff parameters for sindy
[sindyCovariances.useTVRegDiff] = deal(0);
[sindyCovariances.iter] = deal(sindy.iter); 
[sindyCovariances.alph] = deal(sindy.alph);
[sindyCovariances.ep] = deal(sindy.ep);
[sindyCovariances.scale] = deal(sindy.scale);
[sindyCovariances.plotflag] = deal(0);
[sindyCovariances.diagflag] = deal(0);

% paths needed for utilizing chris's library, add the paths for your system and restart matlab if matlab cant find them
p = py.sys.path;
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\')
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\advlearn\attacks\poison\')
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\advlearn\attacks\')

% SVMAttack Parameters
% Setup Boundary Regions
classNumber = 1; % class to attack
mesh = 10.5
step = 0.25;
bound = mesh-0.5;
boundary = [[-bound -bound];[bound bound]];
svmPoisonAttackArgs = pyargs('boundary', boundary,...
							 'step_size', 0.5,...
							  'max_steps', int32(100),...
							  'c', int32(classNumber);,...
							  'kernel', 'rbf',...
							  'degree', 3,...
							   'coef0', 1,...
							   'gamma', 1);



% run learn++.nse, and attack
n_timestamps = length(data_train);  % total number of time stamps
f_measure = zeros(n_timestamps, net.mclass);
g_mean = zeros(n_timestamps, 1);
recall = zeros(n_timestamps, net.mclass);
precision = zeros(n_timestamps, net.mclass);
err_nse = zeros(n_timestamps, 1);
for ell = 1:n_timestamps
	if ell == 1 % Wait two time steps before making preditions with SINDy and attacking
		[~,...
		f_measure(ell,:),...
		g_mean(ell),...
		precision(ell,:),...
		recall(ell,:),...
		errs_nse(ell)] = learn_nse_for_attacking(net,...
												 data_train{ell},...
												 labels_train{ell},...
												 data_test{ell},...
		                                         labels_test{ell});
	else 
		meansData = cell2mat({[classMeans{classNumber}{1,1:n_timestamps}]});
		meansData = reshape(meansData',n_timestamps,numDims)';
		sindyMeans(classNumber).buildModel(meansData,1,1,n_timestamps,1);
		
		[attackPoints,attackLabels] = chrisAttacks(data,labels,boundary,svmPoisonAttackArgs,numberAttackPoints);
		[~,...
		f_measure(ell,:),...
		g_mean(ell),...
		precision(ell,:),...
		recall(ell,:),...
		errs_nse(ell)] = learn_nse_for_attacking(net,...
												 data_train{ell},...
												 labels_train{ell},...
												 data_test{ell},...
		                                         labels_test{ell});

end
