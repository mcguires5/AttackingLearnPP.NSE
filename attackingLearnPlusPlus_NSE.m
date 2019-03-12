%% Add paths
addpath(fullfile('.','SINDy'));
addpath(fullfile('.','Learn++NSE'));
addpath(fullfile('.','ConceptDriftData'));
addpath(genpath(fullfile('.','advlearn')));
%% Generate data or load data??
T = 200;  % number of time stamps
N = 100;  % number of data points at each time
[data_train, labels_train,data_test,labels_test] = ConceptDriftData('sea', T, N);
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

%% Classifier parameters
model.type = 'SVM';          % base classifier
net.a = .5;                   % slope parameter to a sigmoid
net.b = 10;                   % cutoff parameter to a sigmoid
net.threshold = 0.01;         % how small is too small for error
net.mclass = 2;               % number of classes in the prediciton problem
net.base_classifier = model;  % set the base classifier in the net struct

%% These cell arrays are structured as {class}{1,timeStep}
numTimeSteps = length(data_train);
[numObs, numDims] = size(data_train{1});
numClasses = numel(unique(labels_train{1}));
classLabelsIdxs = cell(1,numClasses);
classData = cell(numTimeSteps,numClasses);
classMeans = cell(numTimeSteps,numClasses);
classVariances = cell(numTimeSteps,numClasses);
for iClass = 1:numClasses
	classLabelsIdxs{iClass} = cellfun(@(x) eq(x,iClass),labels_train,'UniformOutput',false); % Get indexes for data points of classNumber at each time step
	classData{iClass} = cellfun(@(a,b) a(b,:),data_train,classLabelsIdxs{iClass},'UniformOutput',false); % Now extract the class data using logical indexing at each time step 
	classMeans(:,iClass) = cellfun(@(x) mean(x),classData{iClass},'UniformOutput',false); % Calculate the means at every time step
	classVariances(:,iClass) = cellfun(@(x) cov(x),classData{iClass},'UniformOutput',false); % Calculate the covariance at every time step
end
%% Create a SINDy model for means, 1 Obj that will be continuously updated.
sindyMeans = cell(1,numClasses); %{Class}(SINDy Obj)
for iClass = 1:numClasses
	sindyMeans{iClass} = SINDy(); % SINDy Object ... type help SINDy for list of what each parameter does
	sindyMeans{iClass}.lambda = 2e-6;
	sindyMeans{iClass}.polyOrder = 5;
	sindyMeans{iClass}.useSine = 0;
	sindyMeans{iClass}.sineMultiplier = 10;
	sindyMeans{iClass}.useExp = 0;
	sindyMeans{iClass}.expMultiplier = 10;
	sindyMeans{iClass}.useCustomPoolData = 1;
	sindyMeans{iClass}.nonDynamical = 0;
	% TVRegDiff parameters for SINDy 
	sindyMeans{iClass}.useTVRegDiff = 0;
	sindyMeans{iClass}.iter = sindyMeans{iClass}.iter; 
	sindyMeans{iClass}.alph = sindyMeans{iClass}.alph;
	sindyMeans{iClass}.ep = sindyMeans{iClass}.ep;
	sindyMeans{iClass}.scale = sindyMeans{iClass}.scale;
	sindyMeans{iClass}.plotflag = 0;
	sindyMeans{iClass}.diagflag = 0;
end
% Create an array of SINDy objects, the array should be equal in size to the covariance of the data, so 2 dim data is 2x2 Covariance matrix, 1 SINDy Obj for each
sindyCovariances = cell(1,numClasses); %{Class}(Array of SINDy Objs, 1 for each value of covariance)
[heightCovMat, widthCovMat] = size(classVariances{1,1}); % Get size of Cov Matrix
for iClass = 1:numClasses
	sindyCovariances{iClass}(heightCovMat,widthCovMat) = SINDy(); % Create the array of SINDy Objects
	% Using the below synatx [objArray.property] = deal(value); will set the property of each object in objArray to value.
	[sindyCovariances{iClass}.lambda] = deal(2e-6);
	[sindyCovariances{iClass}.polyOrder] = deal(5);
	[sindyCovariances{iClass}.useSine] = deal(0);
	[sindyCovariances{iClass}.sineMultiplier] = deal(10);
	[sindyCovariances{iClass}.useExp] = deal(0);
	[sindyCovariances{iClass}.expMultiplier] = deal(10);
	[sindyCovariances{iClass}.useCustomPoolData] = deal(1);
	[sindyCovariances{iClass}.nonDynamical] = deal(0);
	% TVRegDiff parameters for sindy
	[sindyCovariances{iClass}.useTVRegDiff] = deal(0);
	[sindyCovariances{iClass}.iter] = deal(sindyCovariances{iClass}.iter); 
	[sindyCovariances{iClass}.alph] = deal(sindyCovariances{iClass}.alph);
	[sindyCovariances{iClass}.ep] = deal(sindyCovariances{iClass}.ep);
	[sindyCovariances{iClass}.scale] = deal(sindyCovariances{iClass}.scale);
	[sindyCovariances{iClass}.plotflag] = deal(0);
	[sindyCovariances{iClass}.diagflag] = deal(0);
end

%% paths needed for utilizing chris's library, add the paths for your system and restart matlab if matlab cant find them
p = py.sys.path;
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\')
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\advlearn\attacks\poison\')
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\advlearn\attacks\')

%% SVMAttack Parameters
% Setup Boundary Regions
% need to set this up for any dataset
classNumber = 1; % class to attack
step_size = 0.5;
max_steps = 100;
c = classNumber;
kernel = 'linear';
degree = 3;
coef0 = 1;
gamma = 1;
numberAttackPoints = 3;
% Mesh and step is for creating boundary of attack
mesh = 10.5;
step = 0.25;
attackPoints = cell(1,numTimeSteps);
attackPoints(:) = {zeros(numberAttackPoints,numDims)};
attackLabels = cell(1,numTimeSteps);
attackLabels(:) = {zeros(numberAttackPoints,1)};
bound = mesh-0.5;
boundary = [[-bound -bound];[bound bound]];
np = py.importlib.import_module('numpy');
boundary = np.array(boundary);
svmPoisonAttackArgs = pyargs('boundary', boundary,...
							 'step_size', step_size,...
							  'max_steps', int32(max_steps),...
							  'c', int32(classNumber),...
							  'kernel', kernel,...
							  'degree', degree,...
							   'coef0', coef0,...
							   'gamma', gamma);



%% run learn++.nse, and attack
n_timestamps = length(data_train);  % total number of time stamps
f_measure = zeros(n_timestamps, net.mclass);
g_mean = zeros(n_timestamps, 1);
recall = zeros(n_timestamps, net.mclass);
precision = zeros(n_timestamps, net.mclass);
errs_nse = zeros(n_timestamps, 1);
generatedData = cell(iClass,numTimeSteps);
generatedData(:) = {zeros(N/2,numDims)};
generatedLabels = cell(iClass,numTimeSteps);
generatedLabels(:) = {zeros(N/2,1)};
mu = cell(numClasses,numTimeSteps);
mu(:) = {zeros(1,numDims)};
sigma = cell(numClasses,numTimeSteps);
sigma(:) = {zeros(heightCovMat,widthCovMat)};
% Does covariance matrix indexing
horzCovLinIdx = 1:widthCovMat;
idxOfNxtRow = horzCovLinIdx(end)+1;	
rowDiff = idxOfNxtRow - horzCovLinIdx(1,1);
covIdx = repmat(horzCovLinIdx,heightCovMat,1);
for iRow = 2:heightCovMat
    covIdx(iRow,:) = covIdx(iRow,:)+(iRow-1)*rowDiff;
end
%[I,J] = ind2sub([heightCovMat widthCovMat],covIdx);
%covSubs = [I(:) J(:)];
covSubs = repmat(covIdx,numTimeSteps,1);
for iTStep = 1:numTimeSteps
	if iTStep == 1 % Wait two time steps before making preditions with SINDy and attacking
		[~,...
		f_measure(iTStep,:),...
		g_mean(iTStep),...
		precision(iTStep,:),...
		recall(iTStep,:),...
		errs_nse(iTStep)] = learn_nse_for_attacking(net,...
												 data_train{iTStep},...
												 labels_train{iTStep},...
												 data_test{iTStep},...
		                                         labels_test{iTStep});
    elseif iTStep == 4
        [~,...
		f_measure(iTStep,:),...
		g_mean(iTStep),...
		precision(iTStep,:),...
		recall(iTStep,:),...
		errs_nse(iTStep)] = learn_nse_for_attacking(net,...
												 data_train{iTStep},...
												 labels_train{iTStep},...
												 data_test{iTStep},...
		                                         labels_test{iTStep});
		for iClass = 1:numClasses
			means = cell2mat(classMeans(1:iTStep,iClass));
			sindyMeans{iClass}.buildModel(means,1,1,iTStep,1); % data,dt,startTime,endTime,numTimeStepsToPredict,<optionally derivatives>
			mu{iClass,iTStep+1} = sindyMeans{iClass}.model(end,:);
			covariance = cell2mat(classVariances(1:iTStep,iClass)); 
			for iCov = 1:numel(covIdx)
                idx = covSubs(1:heightCovMat*iTStep,1:widthCovMat) == iCov;
				sindyCovariances{iClass}(covIdx == iCov).buildModel(covariance(idx),1,1,iTStep,1);
				sigma{iClass,iTStep+1}(covIdx == iCov) = ...
                    sindyCovariances{iClass}(covIdx == iCov).model(end,1);
            end
            generatedData{iClass,iTStep+1} = mvnrnd(mu{iClass,iTStep+1},sigma{iClass,iTStep+1},N/2);
            generatedLabels{iClass,iTStep+1} = repmat(iClass,N/2,1);
        end
        [attackPoints{iTStep+1},attackLabels{iTStep+1}] = ...
            chrisAttacks(generatedData{iClass,iTStep+1},...
                         generatedLabels{iClass,iTStep+1},...
                         boundary,svmPoisonAttackArgs,numberAttackPoints);

		% need to do sindyModels for covariances, and then generate attacks
		[~,...
		f_measure(iTStep,:),...
		g_mean(iTStep),...
		precision(iTStep,:),...
		recall(iTStep,:),...
		errs_nse(iTStep)] = learn_nse_for_attacking(net,...
												 [data_train{iTStep};attackPoints{iTStep}],...
												 [labels_train{iTStep};attackPoints{iTStep}],...
												 data_test{iTStep},...
		                                         labels_test{iTStep});
    elseif iTStep >= 5 
     for iClass = 1:numClasses
			means = cell2mat(classMeans(1:iTStep,iClass));
			sindyMeans{iClass}.buildModel(means,1,1,iTStep,1); % data,dt,startTime,endTime,numTimeStepsToPredict,<optionally derivatives>
			mu{iClass,iTStep+1} = sindyMeans{iClass}.model(end,:);
			covariance = cell2mat(classVariances(1:iTStep,iClass)); 
			for iCov = 1:numel(covIdx)
                idx = covSubs(1:heightCovMat*iTStep,1:widthCovMat) == iCov;
				sindyCovariances{iClass}(covIdx == iCov).buildModel(covariance(idx),1,1,iTStep,1);
				sigma{iClass,iTStep+1}(covIdx == iCov) = ...
                    sindyCovariances{iClass}(covIdx == iCov).model(end,1);
            end
            generatedData{iClass,iTStep+1} = mvnrnd(mu{iClass,iTStep},sigma{iClass,iTStep},N/2);
            generatedLabels{iClass,iTStep+1} = repmat(iClass,N/2,1);
        end
        [attackPoints{iTStep+1},attackLabels{iTStep+1}] = ...
            chrisAttacks(generatedData{iClass,iTStep+1},...
                         generatedLabels{iClass,iTStep+1},...
                         boundary,svmPoisonAttackArgs,numberAttackPoints);
    end
end