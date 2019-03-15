%% Add paths
clearvars
close all
addpath(fullfile('.','SINDy'));
addpath(fullfile('.','Learn++NSE'));
addpath(fullfile('.','ConceptDriftData'));
addpath(genpath(fullfile('.','advlearn')));
%% Generate data or load data??
%T = 200;  % number of time stamps
%N = 100;  % number of data points at each time
%[data_train, labels_train,data_test,labels_test] = ConceptDriftData('sea', T, N);
dataset = "X2CDT";
x = load("Synthetic Datasets\" + dataset + ".mat");
data_train = x.(dataset).train_data;
data_test = x.(dataset).test_data;
labels_test = x.(dataset).test_labels;
labels_train = x.(dataset).train_labels;
T = length(data_train);
N = x.(dataset).drift;
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
%% These cell arrays are structured as {class}{1,timeStep}
numTimeSteps = length(data_train);
[numObs, numDims] = size(data_train{1});
numClasses = numel(unique(labels_train{1}));
classLabelsIdxs = cell(1,numClasses);
classData = cell(1,numClasses);
classMeans = cell(numTimeSteps,numClasses);
classVariances = cell(numTimeSteps,numClasses);
numObsPerClass = cell(numTimeSteps,numClasses);
for iClass = 1:numClasses
	classLabelsIdxs{iClass} = cellfun(@(x) eq(x,iClass),labels_train,...
                              'UniformOutput',false); % Get indexes for data points of classNumber at each time step
	classData{iClass} = cellfun(@(a,b) a(b,:),data_train,classLabelsIdxs{iClass},...
                        'UniformOutput',false); % Now extract the class data using logical indexing at each time step 
	classMeans(:,iClass) = cellfun(@(x) mean(x),classData{iClass},...
                           'UniformOutput',false); % Calculate the means at every time step
	classVariances(:,iClass) = cellfun(@(x) cov(x),classData{iClass},...
                               'UniformOutput',false); % Calculate the covariance at every time step
    numObsPerClass(:,iClass) = cellfun(@(x) size(x,1), classData{iClass},...
                               'UniformOutput',false);
end
%% structure array of variables and stuff, each row is a time step
nseData = repmat(struct("dataTrain",zeros(numObs,numDims),...
                        "dataTest",zeros(numObs,numDims),...
                        "labelsTrain",zeros(numObs,1),...
                        "labelsTest",zeros(numObs,1)),1,numTimeSteps);
[nseData.dataTrain] = data_train{:};
[nseData.dataTest] = data_test{:};
[nseData.labelsTrain] = labels_train{:};
[nseData.labelsTest] = labels_test{:};
for iClass = 1:numClasses
    SINDyData.("muC"+iClass) = 0;
    SINDyData.("sigmaC"+iClass) = 0;
    SINDyData.("dataC"+iClass) = 0;
    SINDyData.("numObsC"+iClass) = 0;
    SINDyData.("idxC"+iClass) = 0;
end
SINDyData = repmat(SINDyData,1,numTimeSteps);
for iClass = 1:numClasses
    [SINDyData.("muC"+iClass)] = classMeans{:,iClass};
    [SINDyData.("sigmaC"+iClass)] = classVariances{:,iClass};
    [SINDyData.("dataC"+iClass)] = classData{iClass}{:};
    [SINDyData.("numObsC"+iClass)] = numObsPerClass{:,iClass};
    [SINDyData.("idxC"+iClass)] = classLabelsIdxs{iClass}{:};
end
clearvars -except numTimeSteps numObs numDims nseData numClasses dataset SINDyData
%% Classifier parameters
model.type = 'SVM';          % base classifier
net.a = .5;                   % slope parameter to a sigmoid
net.b = 10;                   % cutoff parameter to a sigmoid
net.threshold = 0.01;         % how small is too small for error
net.mclass = 2;               % number of classes in the prediciton problem
net.base_classifier = model;  % set the base classifier in the net struct
%% Create a SINDy model for means, 1 Obj that will be continuously updated.
SINDy(1:numTimeSteps,1:numClasses) = SINDy(); % SINDy Object ... type help SINDy for list of what each parameter does
[SINDy.lambda] = deal(2e-6);
[SINDy.polyOrder] = deal(2);
[SINDy.useSine] = deal(0);
[SINDy.sineMultiplier] = deal(10);
[SINDy.useExp] = deal(0);
[SINDy.expMultiplier] = deal(10);
[SINDy.useCustomPoolData] = deal(1);
[SINDy.nonDynamical] = deal(0);
% TVRegDiff parameters for SINDy 
[SINDy.useTVRegDiff] = deal(0);
[SINDy.iter] = deal(10); 
[SINDy.alph] = deal(0.00002);
[SINDy.ep] = deal(1e12);
[SINDy.scale] = deal("small");
[SINDy.plotflag] = deal(0);
[SINDy.diagflag] = deal(0);
%% paths needed for utilizing chris's library, add the paths for your system and restart matlab if matlab cant find them
p = py.sys.path;
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\')
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\advlearn\attacks\poison\')
insert(p,int32(0),'H:\Gits\AttackingLearnPP.NSE\advlearn\advlearn\attacks\')

%% SVMAttack Parameters
% Setup Boundary Regions
% need to set this up for any dataset
atkSet.step_size = 0.5;
atkSet.timeToAttack = 4;
atkSet.max_steps = 100;
atkSet.c = 1;
atkSet.kernel = 'linear';
atkSet.degree = 3;
atkSet.coef0 = 1;
atkSet.gamma = 1;
atkSet.numAtkPts = 3;
% Mesh and step is for creating boundary of attack
atkSet.mesh = 10.5;
atkSet.step = 0.25;
% variables to hold attack points and generated data sets
SINDyGen = repmat(struct("data",zeros(numObs,numDims),"labels",zeros(numObs,1)),1,numTimeSteps);
atkData = repmat(struct("points",zeros(atkSet.numAtkPts,numDims),...
                           "labels",zeros(atkSet.numAtkPts,1)),1,numTimeSteps);
[SINDyGen(1:atkSet.timeToAttack).data] = deal(0);
[SINDyGen(1:atkSet.timeToAttack).labels] = deal(0);
[atkData(1:atkSet.timeToAttack).points] = deal(0);
[atkData(1:atkSet.timeToAttack).labels] = deal(0);
np = py.importlib.import_module('numpy');
boundary = [[-bound -bound];[bound bound]];
atkSet.boundary = np.array(boundary);
svmPoisonAttackArgs = pyargs('boundary', atkSet.boundary,...
							 'step_size', atkSet.step_size,...
							  'max_steps', int32(atkSet.max_steps),...
							  'c', int32(atkSet.c),...
							  'kernel', atkSet.kernel,...
							  'degree', atkSet.degree,...
							   'coef0', atkSet.coef0,...
							   'gamma', atkSet.gamma);
%% run learn++.nse, and attack
nseResults = repmat(struct("f_measure",zeros(1,net.mclass),...
                              "g_mean",0,...
                              "recall",zeros(1,net.mclass),...
                              "precision",zeros(1,net.mclass),...
                              "errs_nse",0),1,numTimeSteps);
for iTStep = 1:numTimeSteps
	if iTStep < atkSet.timeToAttack % Wait time steps before making preditions with SINDy and attacking
		[~,...
		nseResults(iTStep).f_measure,...
		nseResults(iTStep).g_mean,...
		nseResults(iTStep).precision,...
		nseResults(iTStep).recall,...
		nseResults(iTStep).errs_nse] = learn_nse_for_attacking(net,...
                                            nseData(iTStep).dataTrain,...
                                            nseData(iTStep).labelsTrain,...
                                            nseData(iTStep).dataTest,...
                                            nseData(iTStep).labelsTrain);
    elseif iTStep == atkSet.timeToAttack
		[~,...
		nseResults(iTStep).f_measure,...
		nseResults(iTStep).g_mean,...
		nseResults(iTStep).precision,...
		nseResults(iTStep).recall,...
		nseResults(iTStep).errs_nse] = learn_nse_for_attacking(net,...
                                            nseData(iTStep).dataTrain,...
                                            nseData(iTStep).labelsTrain,...
                                            nseData(iTStep).dataTest,...
                                            nseData(iTStep).labelsTrain);
		for iClass = 1:numClasses
			SINDy(iTStep,iClass).buildModel(vertcat(SINDyData(1:iTStep).("muC"+iClass)),1,1,iTStep,1); % data,dt,startTime,endTime,numTimeStepsToPredict,<optionally derivatives>
            SINDyGen(iTStep+1).data(SINDyData(iTStep).("idxC"+iClass),:) = ...
                                                    mvnrnd(SINDy(iTStep,iClass).model(end,:),...
                                                           SINDyData(iTStep).("sigmaC"+iClass),...
                                                           SINDyData(iTStep).("numObsC"+iClass));
            SINDyGen(iTStep+1).labels(SINDyData(iTStep).("idxC"+iClass)) = ...
                                                    repmat(iClass,SINDyData(iTStep).("numObsC"+iClass),1);
        end
        % Need to dynamically determine bounds for an arbitrary dimensions
        % of data. 
        [atkData(iTStep+1).data,atkData(iTStep+1).labels] = ...
            chrisAttacks(SINDyGen(iTStep+1).data,...
                         SINDyGen(iTStep+1).labels,...
                         boundary,svmPoisonAttackArgs,numberAttackPoints);
    elseif iTStep > atkSet.timeToAttack 
		[~,...
		nseResults(iTStep).f_measure,...
		nseResults(iTStep).g_mean,...
		nseResults(iTStep).precision,...
		nseResults(iTStep).recall,...
		nseResults(iTStep).errs_nse] = learn_nse_for_attacking(net,...
                                            nseData(iTStep).dataTrain,...
                                            nseData(iTStep).labelsTrain,...
                                            nseData(iTStep).dataTest,...
                                            nseData(iTStep).labelsTrain);

    end
end
function [] = DummyFunctionForHoldingCovarianceCode()
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

    covariance = cell2mat(classVariances(1:iTStep,iClass)); 
    for iCov = 1:numel(covIdx)
        idx = covSubs(1:heightCovMat*iTStep,1:widthCovMat) == iCov;
        sindyCovariances{iClass}(covIdx == iCov).buildModel(covariance(idx),1,1,iTStep,1);
        sigma{iClass,iTStep+1}(covIdx == iCov) = ...
            sindyCovariances{iClass}(covIdx == iCov).model(end,1);
    end

end