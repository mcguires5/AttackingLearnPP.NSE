function [classInfo,...
          nseData,nseResults,...
          numClasses,numDims,...
          numObs,numTimeSteps] = learnPlusPlus_NSE_Preprocessing(dataset,kernel)

	x = load("Synthetic Datasets\" + dataset + ".mat");
	data_train = x.(dataset).dataTrain;
	data_test = x.(dataset).dataTest;
	labels_test = x.(dataset).labelsTest;
	labels_train = x.(dataset).labelsTrain;
	T = length(data_train);
	for t = 1:T
	  data_train{t} = data_train{t};
	  labels_train{t} = labels_train{t};
	  data_test{t} = data_test{t};
	  labels_test{t} = labels_test{t};
	end
	% Extracts all the data information
	numTimeSteps = length(data_train);
	[numObs, numDims] = size(data_train{1});
	numClasses = numel(unique(labels_train{1}));
	classLabelsIdxs = cell(numTimeSteps,numClasses);
	classData = cell(numTimeSteps,numClasses);
	classMeans = cell(numTimeSteps,numClasses);
	classVariances = cell(numTimeSteps,numClasses);
	numObsPerClass = cell(numTimeSteps,numClasses);
	for iClass = 1:numClasses
		[classLabelsIdxs(:,iClass)] = cellfun(@(x) eq(x,iClass),labels_train',...
	                              'UniformOutput',false); % Get indexes for data points of classNumber at each time step
		[classData(:,iClass)] = cellfun(@(a,b) a(b,:),data_train',classLabelsIdxs(:,iClass),...
	                        'UniformOutput',false); % Now extract the class data using logical indexing at each time step 
		[classMeans(:,iClass)] = cellfun(@(x) mean(x),classData(:,iClass),...
	                           'UniformOutput',false); % Calculate the means at every time step
		[classVariances(:,iClass)] = cellfun(@(x) cov(x),classData(:,iClass),...
	                               'UniformOutput',false); % Calculate the covariance at every time step
	    [numObsPerClass(:,iClass)] = cellfun(@(x) size(x,1), classData(:,iClass),...
	                               'UniformOutput',false);
	end
	% nseData is structure array (timestep), stores training data, test data, and labels to feed into Learn++.NSE

	nseData = repmat(struct("dataTrain",zeros(numObs,numDims),...
	                        "dataTest",zeros(numObs,numDims),...
	                        "labelsTrain",zeros(numObs,1),...
	                        "labelsTest",zeros(numObs,1)),1,numTimeSteps);
	[nseData.dataTrain] = data_train{:};
	[nseData.dataTest] = data_test{:};
	[nseData.labelsTrain] = labels_train{:};
	[nseData.labelsTest] = labels_test{:};
	% classInfo is structure array (timestep,class), stores data, means, covariances, number of observations, and indexes of classes within data matrix

	classInfo = repmat(struct("mu",zeros(1,numDims),...
                                  "sigma",zeros(numDims,numDims),...
                                  "data",zeros(numObs,numDims),...
                                  "numObs",0,...
                                  "idx",false(numObs,1)),numTimeSteps,numClasses);
	[classInfo(:,:).mu] = classMeans{:,:};
	[classInfo(:,:).sigma] = classVariances{:,:};
	[classInfo(:,:).data] = classData{:,:};
	[classInfo(:,:).numObs] = numObsPerClass{:,:};
	[classInfo(:,:).idx] = classLabelsIdxs{:,:};
	% Classifier parameters
	model.type = 'SVM';          % base classifier
	model.kernel = kernel;
	net.a = .5;                   % slope parameter to a sigmoid
	net.b = 10;                   % cutoff parameter to a sigmoid
	net.threshold = 0.01;         % how small is too small for error
	net.mclass = 2;               % number of classes in the prediciton problem
	net.base_classifier = model;  % set the base classifier in the net struct
	net.initialized = true;
	net.classifiers = {};   % classifiers
	net.w = [];             % weights 
	net.initialized = false;% set to false
	net.t = 1;              % track the time of learning
	net.classifierweigths = {};               % array of classifier weights
	net.type = 'learn++.nse';
	% run learn++.nse, and attack
	nseResults = repmat(struct("net",net,...
	                              "f_measure",zeros(1,net.mclass),...
	                              "g_mean",0,...
	                              "recall",zeros(1,net.mclass),...
	                              "precision",zeros(1,net.mclass),...
	                              "errs_nse",0,...
	                              "latestClassifier",0),1,numTimeSteps);	
end