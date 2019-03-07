% Generate data or load data??
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

% Classifier parameters
model.type = 'SVM';          % base classifier
net.a = .5;                   % slope parameter to a sigmoid
net.b = 10;                   % cutoff parameter to a sigmoid
net.threshold = 0.01;         % how small is too small for error
net.mclass = 2;               % number of classes in the prediciton problem
net.base_classifier = model;  % set the base classifier in the net struct

classNumber = 1; % class to attack and build SINDy models for


% These cell arrays are structured as {1,timeStep}
numTimeSteps = length(data_train);
[numObs numDims] = size(data_train{1});
classLabelsIdxs = cell(1,numTimeSteps);
classData = cell(1,numTimeSteps);
classMeans = cell(1,numTimeSteps);
classVariances = cell(1,numTimeSteps);
classLabelsIdxs = cellfun(@(x) eq(x,classNumber),labels_train,'UniformOutput',false); % Get indexes for data points of classNumber at each time step
classData = cellfun(@(a,b) a(b,:),data_train,classLabelsIdxs,'UniformOutput',false); % Now extract the class data using logical indexing at each time step 
classMeans = cellfun(@(x) mean(x),classData,'UniformOutput',false); % Calculate the means at every time step
classVariances = cellfun(@(x) cov(x),classData,'UniformOutput',false); % Calculate the covariance at every time step

% Create a SINDy model for means, 1 Obj that will be continuously updated.
sindyMeans = SINDy(); % SINDy Object ... type help SINDy for list of what each parameter does
sindyMeans.lambda = 2e-6;
sindyMeans.polyOrder = 5;
sindyMeans.useSine = 0;
sindyMeans.sineMultiplier = 10;
sindyMeans.useExp = 0;
sindyMeans.expMultiplier = 10;
sindyMeans.useCustomPoolData = 1;
sindyMeans.nonDynamical = 0;
% TVRegDiff parameters for SINDy 
sindyMeans.useTVRegDiff = 0;
sindyMeans.iter = sindyMeans.iter; 
sindyMeans.alph = sindyMeans.alph;
sindyMeans.ep = sindyMeans.ep;
sindyMeans.scale = sindyMeans.scale;
sindyMeans.plotflag = 0;
sindyMeans.diagflag = 0;
% Create an array of SINDy objects, the array should be equal in size to the covariance of the data, so 2 dim data is 2x2 Covariance matrix, 1 SINDy Obj for each
[heightCovMat, widthCovMat] = size(classVariances{1}); % Get size of Cov Matrix
sindyCovariances(heightCovMat,widthCovMat) = SINDy(); % Create the array of SINDy Objects
% Using the below synatx [objArray.property] = deal(value); will set the property of each object in objArray to value.
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
[sindyCovariances.iter] = deal(sindyCovariances.iter); 
[sindyCovariances.alph] = deal(sindyCovariances.alph);
[sindyCovariances.ep] = deal(sindyCovariances.ep);
[sindyCovariances.scale] = deal(sindyCovariances.scale);
[sindyCovariances.plotflag] = deal(0);
[sindyCovariances.diagflag] = deal(0);

% paths needed for utilizing chris's library, add the paths for your system and restart matlab if matlab cant find them
p = py.sys.path;
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\')
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\advlearn\attacks\poison\')
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\advlearn\attacks\')

% SVMAttack Parameters
% Setup Boundary Regions
% need to set this up for any dataset
step_size = 0.5;
max_steps = 100;
c = classNumber;
kernel = 'linear';
degree = 3;
coef0 = 1;
gamma = 1;

mesh = 10.5;
step = 0.25;
bound = mesh-0.5;
boundary = [[-bound -bound];[bound bound]];
svmPoisonAttackArgs = pyargs('boundary', boundary,...
							 'step_size', step_size,...
							  'max_steps', int32(max_steps),...
							  'c', int32(classNumber),...
							  'kernel', kernel,...
							  'degree', degree,...
							   'coef0', coef0,...
							   'gamma', gamma);



% run learn++.nse, and attack
n_timestamps = length(data_train);  % total number of time stamps
f_measure = zeros(n_timestamps, net.mclass);
g_mean = zeros(n_timestamps, 1);
recall = zeros(n_timestamps, net.mclass);
precision = zeros(n_timestamps, net.mclass);
errs_nse = zeros(n_timestamps, 1);
sigma = zeros(heightCovMat,widthCovMat);
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
	else 
		meansData = cell2mat({[classMeans{1:iTStep}]});
		meansData = reshape(meansData',iTStep,numDims)'; % Reshape into timeStep By numDims matrix
		sindyMeans.buildModel(meansData,1,1,iTStep,1); % data,dt,startTime,endTime,numTimeStepsToPredict,<optionally derivatives>
		mu = sindyMeans.model(end,:);
		covarianceData = cell2mat({[classVariances{1:iTStep}]});
		covarianceData = reshape(covarianceData,heightCovMat,widthCovMat,iTStep);
		horzCovLinIdx = 1:widthCovMat;
		idxOfNxtRow = horzCovLinIdx(end)+1;	
		secRowCovLinIdx = idxOfNxtRow:widthCovMat+idxOfNxtRow-1;
		covIdx = [horzCovLinIdx;secRowCovLinIdx];
		[I,J] = ind2sub([heightCovMat widthCovMat],covIdx);
		covSubs = [I(:) J(:)];
		covSubs2 = repmat(covIdx,1,1,iTStep);
		for iCov = 1:size(covSubs,1)
			idx = covSubs2 == iCov;
			sindyCovariances(idx(:,:,1)).buildModel(covarianceData(idx),1,1,iTStep,1);
			sigma(idx(:,:,1)) = sindyCovariances(idx(:,:,1)).model(end,1);
		end
		generatedData = mvnrnd(mu,sigma,N);


		% need to do sindyModels for covariances, and then generate attacks
		[attackPoints,attackLabels] = chrisAttacks(generatedData,labels,boundary,svmPoisonAttackArgs,numberAttackPoints);
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
    end
end
