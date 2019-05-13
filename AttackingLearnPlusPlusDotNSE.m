% @Author: delengowski
% @Date:   2019-04-30 17:18:49
% @Last Modified by:   delengowski
% @Last Modified time: 2019-05-11 18:17:00
function [NseResults,varargout] = AttackingLearnPlusPlusDotNSE(DataSet,AttackMethod,Kernel,varargin)

	% Create input parser
	parser = inputParser;

	% Input checking functions
	AttackMethods = ["PredictiveOptimized","PredictiveLabelFlipping","NonPredictiveOptimized"];
	AttackMethodMsg = "AttackMethod must be one of the following " + AttackMethods;
	AttackMethodValidation = @(x) assert(ismember(x,AttackMethods),AttackMethodMsg);
	Kernels = ["Linear","RBF"];
	KernelMsg = "Kernel must be one of the following" + Kernels;
	KernelValidation = @(x) assert(ismember(x,Kernels),KernelMsg);
	WindowSizeMsg = "Predictive Attacks must be given at least 3 time steps to generate SINDy models.";
	WindowSizeValidation = @(x) assert(x >= 3,WindowSizeMsg);
    

	% Require inputs
	addRequired(parser,"DataSet")
    addRequired(parser,"AttackMethod",AttackMethodValidation);
    addRequired(parser,"Kernel",KernelValidation);

	% Optional inputs
		% Learn++.NSE settings
	addOptional(parser,"A",.5);
	addOptional(parser,"B",10);
	addOptional(parser,"Threshold",0.01);
		% SINDy settings
		% SINDy object will parse these, no need to check here
	addOptional(parser,"Lambda",2e-6);
	addOptional(parser,"PolyOrder",5);
	addOptional(parser,"UseSine",0);
	addOptional(parser,"SineMultiplier",10);
	addOptional(parser,"UseExp",0);
	addOptional(parser,"ExpMultiplier",10);
    addOptional(parser,"MeasureSINDyResults",0);
		% AdvLearn settings
    addOptional(parser,"MaxSteps",200);
    addOptional(parser,"StepSize",5);
    addOptional(parser,"Degree",3);
    addOptional(parser,"Coef0",1);
    addOptional(parser,"Gamma","auto");
    addOptional(parser,"ClassToAttack",1);
    addOptional(parser,"NumAttackPoints",3);
    addOptional(parser,"WindowSize",3,WindowSizeValidation);
    addOptional(parser,"TimeStepsInPastToAttack",1)
    addOptional(parser,"ReturnPlottingInfo",0);

	% Check inputs
	parse(parser,DataSet,AttackMethod,Kernel,varargin{:});

	% Distribute the optional inputs
        % Learn++.NSE setting
    A = parser.Results.A;
    B = parser.Results.B;
    Threshold = parser.Results.Threshold;
		% SINDy settings
	Lambda = parser.Results.Lambda;
	PolyOrder = parser.Results.PolyOrder;
	UseSine = parser.Results.UseSine;
	SineMultiplier = parser.Results.SineMultiplier;
	UseExp = parser.Results.UseExp;
	ExpMultiplier = parser.Results.ExpMultiplier;
    MeasureSINDyResults = parser.Results.MeasureSINDyResults;
		% AdvLearn settings
	MaxSteps = parser.Results.MaxSteps;
	StepSize = parser.Results.StepSize;
	Degree = parser.Results.Degree;
	Coef0 = parser.Results.Coef0;
	Gamma = parser.Results.Gamma;
	ClassToAttack = parser.Results.ClassToAttack;
	NumAttackPoints = parser.Results.NumAttackPoints;
    WindowSize = parser.Results.WindowSize;
	TimeStepsInPastToAttack = parser.Results.TimeStepsInPastToAttack;
        % General Setting
	ReturnPlottingInfo = parser.Results.ReturnPlottingInfo;
    
    

	% Create variable output argument list
	NumOutArgs = max(nargout,1);
	varargout = cell(1,NumOutArgs);

	% Load dataset
	x = load("Synthetic Datasets\" + DataSet + ".mat");
	DataTrain = x.(DataSet).dataTrain;
	DataTest = x.(DataSet).dataTest;
	LabelsTest = x.(DataSet).labelsTest;
	LabelsTrain = x.(DataSet).labelsTrain;

	% Extract dataset information 
		% Preallocations
	NumTimeSteps = length(DataTrain);
	[NumObs,NumDims] = size(DataTrain{1});
	NumClasses = numel(unique(LabelsTrain{1}));
	ClassLabelsIdxs = cell(NumTimeSteps,NumClasses);
	ClassData = cell(NumTimeSteps,NumClasses);
	ClassMeans = cell(NumTimeSteps,NumClasses);
	ClassVariances = cell(NumTimeSteps,NumClasses);
	NumObsPerClass = cell(NumTimeSteps,NumClasses);
		% Extractions
	for iClass = 1:NumClasses
		% Get indexes for data points of classNumber at each time step
		[ClassLabelsIdxs(:,iClass)] = cellfun(@(x) eq(x,iClass),LabelsTrain','UniformOutput',false); 
		% Now extract the class data using logical indexing at each time step 
		[ClassData(:,iClass)] = cellfun(@(a,b) a(b,:),DataTrain',ClassLabelsIdxs(:,iClass),'UniformOutput',false);
        % Calculate the means at every time step	 
		[ClassMeans(:,iClass)] = cellfun(@(x) mean(x),ClassData(:,iClass),'UniformOutput',false);
        % Calculate the covariance at every time step 
		[ClassVariances(:,iClass)] = cellfun(@(x) cov(x),ClassData(:,iClass),'UniformOutput',false); 
		% Get number of observations per class at every time step
	    [NumObsPerClass(:,iClass)] = cellfun(@(x) size(x,1), ClassData(:,iClass),'UniformOutput',false);
	end

	% Collect data set info into struct
	DataSetInfo.NumTimeSteps = NumTimeSteps;
	DataSetInfo.NumObs = NumObs;
	DataSetInfo.NumDims = NumDims;
	DataSetInfo.NumClasses = NumClasses;
    
    % Distribute Learn++.NSE settings
	model.type = 'SVM';          % base classifier
	model.kernel = Kernel;
	Net.a = A;                   % slope parameter to a sigmoid
	Net.b = B;                   % cutoff parameter to a sigmoid
	Net.threshold = Threshold;         % how small is too small for error
	Net.mclass = 2;               % number of classes in the prediction problem
	Net.base_classifier = model;  % set the base classifier in the net struct
	Net.classifiers = {};   % classifiers
	Net.w = [];             % weights 
	Net.initialized = false; % set to false
	Net.t = 1;               % track the time of learning
	Net.classifierweigths = {}; % array of classifier weights
	Net.type = 'learn++.nse';

	% Preallocation for Learn++.NSE
	% NseData is structure array (row is timestep), stores training data, test data, and labels to feed into Learn++.NSE
	NseData = repmat(struct("DataTrain",zeros(NumObs,NumDims),...
	                        "DataTest",zeros(NumObs,NumDims),...
	                        "LabelsTrain",zeros(NumObs,1),...
	                        "LabelsTest",zeros(NumObs,1)),1,NumTimeSteps);
    [NseData.DataTrain] = DataTrain{:};
	[NseData.DataTest] = DataTest{:};
	[NseData.LabelsTrain] = LabelsTrain{:};
	[NseData.LabelsTest] = LabelsTest{:};
	% NseResults is a structure (row is timestep), stores output of Learn++.NSE function
	NseResults = repmat(struct("Net",Net,...
	                           "FMeasure",zeros(1,NumClasses),...
	                           "GMean",0,...
	                           "Recall",zeros(1,NumClasses),...
	                           "Precision",zeros(1,NumClasses),...
	                           "Error",0,...
	                           "LatestClassifier",0),1,NumTimeSteps);

    % Store class info for feeding into SINDy ClassInfo structure is indexed (timestep,class)
	ClassInfo = repmat(struct("Mu",zeros(1,NumDims),...
                              "Sigma",zeros(NumDims,NumDims),...
                              "Data",zeros(NumObs,NumDims),...
                              "NumObs",0,...
                              "Idx",false(NumObs,1)),NumTimeSteps,NumClasses);
	[ClassInfo(:,:).Mu] = ClassMeans{:,:};
	[ClassInfo(:,:).Sigma] = ClassVariances{:,:};
	[ClassInfo(:,:).Data] = ClassData{:,:};
	[ClassInfo(:,:).NumObs] = NumObsPerClass{:,:};
	[ClassInfo(:,:).Idx] = ClassLabelsIdxs{:,:};	


	% SINDy PREPROCESSING
	if (AttackMethod == "PredictiveOptimized" || AttackMethod == "PredictiveLabelFlipping")

	% SINDy data preallocation 
	SINDyData = repmat(struct("Mu",zeros(1,NumDims),"Sigma",zeros(NumDims,NumDims)),NumTimeSteps,NumClasses);

	TimeVector = (1:NumTimeSteps)';
	CentroidsWithinWindow = [TimeVector,TimeVector+WindowSize];
	CentroidsWithinWindow(WindowSize:end,1) = CentroidsWithinWindow(WindowSize:end,1) - WindowSize + 1; 
	CentroidsWithinWindow(WindowSize:end,2) = CentroidsWithinWindow(WindowSize:end,2) - WindowSize; 

	% Distribute relevant data to SINDyData
	for iClass = 1:NumClasses
		for iTStep = 2:NumTimeSteps
			idx1 = CentroidsWithinWindow(iTStep,1);
			idx2 = CentroidsWithinWindow(iTStep,2);
 			SINDyData(iTStep,iClass).Mu = vertcat(ClassInfo(idx1:idx2,iClass).Mu);
 			SINDyData(iTStep,iClass).Sigma = ClassInfo(iTStep-1,iClass).Sigma;
		end
	end
	[SINDyData.LatestReset] = deal(1);

	% SINDyResults is a structure array (timestep,class), stores the predicted mean and table of learned functions 
	SINDyResults = repmat(struct("Prediction",0,"LearnedFunctions",0),NumTimeSteps,NumClasses);

	% Create SINDy objects
	for iClass = NumClasses:-1:1
		% SINDy Object ... type help SINDy for list of what each parameter does
	    SINDyArray(1,NumClasses) = SINDy();
	end

	% Distribute SINDy settings
	[SINDyArray.lambda] = deal(Lambda);
	[SINDyArray.polyOrder] = deal(PolyOrder);
	[SINDyArray.useSine] = deal(UseSine);
	[SINDyArray.sineMultiplier] = deal(SineMultiplier);
	[SINDyArray.useExp] = deal(UseExp);
	[SINDyArray.expMultiplier] = deal(ExpMultiplier);
	[SINDyArray.useCustomPoolData] = deal(1);
	[SINDyArray.nonDynamical] = deal(0);
	% TVRegDiff parameters for SINDy 
	[SINDyArray.useTVRegDiff] = deal(0);
	[SINDyArray.iter] = deal(10); 
	[SINDyArray.alph] = deal(0.00002);
	[SINDyArray.ep] = deal(1e12);
	[SINDyArray.scale] = deal("small");
	[SINDyArray.plotflag] = deal(0);
	[SINDyArray.diagflag] = deal(0);

	end
	% END SINDy PREPROCESSING

	% Distribute AdvLearn settings
	if (AttackMethod == "PredictiveOptimized" || AttackMethod == "NonPredictiveOptimized")
    
    % Load modules as globals
    global np;
    global Poison;
    
    np = py.importlib.import_module('numpy');
    Poison = py.importlib.import_module('advlearn.attacks.poison'); 
    
    
	AtkSettings.StepSize = StepSize;
	AtkSettings.MaxSteps = MaxSteps;
	AtkSettings.ClassToAttack = ClassToAttack;
	AtkSettings.Kernel = lower(Kernel);
	AtkSettings.Degree = Degree;
	AtkSettings.Coef0 = Coef0;
	AtkSettings.Gamma = Gamma;
	AtkSettings.NumAtkPts = NumAttackPoints;

	end

	% AdvLearn for Predictive Attacks PREPROCESING
	
	if (AttackMethod == "PredictiveOptimized")

	GenDistr = repmat(struct("Data",zeros(NumObs,NumDims),"Labels",zeros(NumObs,1)),1,NumTimeSteps);
	AtkData = repmat(struct("Points",zeros(AtkSettings.NumAtkPts,NumDims),...
	                           "Labels",zeros(1,AtkSettings.NumAtkPts)),1,NumTimeSteps);
    % Zero out distributions and attack data where we do not attack
	[GenDistr(1:WindowSize).Data] = deal(0);
	[GenDistr(1:WindowSize).Labels] = deal(0);
	[AtkData(1:WindowSize).Points] = deal(0);
	[AtkData(1:WindowSize).Labels] = deal(0);

	end

	% END AdvLearn predictive attacks PREPROCESSING

	if (AttackMethod == "NonPredictiveOptimized")

	AtkData = repmat(struct("Points",zeros(AtkSettings.NumAtkPts,NumDims),...
	                        "Labels",zeros(AtkSettings.NumAtkPts,1)),1,NumTimeSteps);
	[AtkData(1:TimeStepsInPastToAttack).Points] = deal(0);
	[AtkData(1:TimeStepsInPastToAttack).Labels] = deal(0);	

    end
    

	if (AttackMethod == "PredictiveOptimized")
	% MAIN CODE for Predictive-Optimized Attacks
	for iTStep = 1:NumTimeSteps
		% dont start attacking until timestep before time to attack
        if (iTStep >= WindowSize) 
            for iClass = 1:NumClasses
                [model, ~,SINDyResults(iTStep,iClass).LearnedFunctions] = ...
                SINDyArray(iClass).buildModel(SINDyData(iTStep,iClass).Mu,...
                                                     1,...
                                                     1,...
                                                     WindowSize,...
                                                     1);
                SINDyResults(iTStep,iClass).Prediction = model(end,:);
                % Generate distribution for next time step
                % last row of SINDy.model is the predicted time step
                GenDistr(iTStep+1).Data(ClassInfo(iTStep,iClass).Idx,:) = ...
                                   mvnrnd(SINDyResults(iTStep,iClass).Prediction,...
                                          SINDyData(iTStep,iClass).Sigma,...
                                          ClassInfo(iTStep,iClass).NumObs);
                GenDistr(iTStep+1).Labels(ClassInfo(iTStep,iClass).Idx,:) = ...
                              repmat(iClass,ClassInfo(iTStep,iClass).NumObs,1);
            end
        end
        if (iTStep <= WindowSize) % perform Learn++.NSE 
            if iTStep == 1
                [NseResults(iTStep).Net,...
                NseResults(iTStep).FMeasure,...
                NseResults(iTStep).GMean,...
                NseResults(iTStep).Precision,...
                NseResults(iTStep).Recall,...
                NseResults(iTStep).Error,...
                NseResults(iTStep).LatestClassifier] = learn_nse_for_attacking(NseResults(iTStep).Net,...
                                                    NseData(iTStep).DataTrain,...
                                                    NseData(iTStep).LabelsTrain,...
                                                    NseData(iTStep).DataTest,...
                                                    NseData(iTStep).LabelsTest);
            else
                [NseResults(iTStep).Net,...
                NseResults(iTStep).FMeasure,...
                NseResults(iTStep).GMean,...
                NseResults(iTStep).Precision,...
                NseResults(iTStep).Recall,...
                NseResults(iTStep).Error,...
                NseResults(iTStep).LatestClassifier] = learn_nse_for_attacking(NseResults(iTStep-1).Net,...
                                                    NseData(iTStep).DataTrain,...
                                                    NseData(iTStep).LabelsTrain,...
                                                    NseData(iTStep).DataTest,...
                                                    NseData(iTStep).LabelsTest);
            end
        elseif (iTStep > WindowSize) % perform Learn++.NSE with attack points added

            % need to construct boundary for n dimensional dataset
            AtkSettings.Boundary = [min(GenDistr(iTStep).Data);max(GenDistr(iTStep).Data)];
            [AtkData(iTStep).Points,AtkData(iTStep).Labels] = ...
                                        chrisAttacks(GenDistr(iTStep).Data,...
                                                     GenDistr(iTStep).Labels,...
                                                     AtkSettings);
            [NseResults(iTStep).Net,...
            NseResults(iTStep).FMeasure,...
            NseResults(iTStep).GMean,...
            NseResults(iTStep).Precision,...
            NseResults(iTStep).Recall,...
            NseResults(iTStep).Error,...
            NseResults(iTStep).LatestClassifier] = ...
            learn_nse_for_attacking(NseResults(iTStep-1).Net,...
                                    [NseData(iTStep).DataTrain;AtkData(iTStep).Points],...
                                    [NseData(iTStep).LabelsTrain;AtkData(iTStep).Labels'],...
                                    NseData(iTStep).DataTest,...
                                    NseData(iTStep).LabelsTest);
            if MeasureSINDyResults
            % Quantify accuracy of SINDy with KL Divergence
            for iClass = 1:NumClasses
                Idx = ClassInfo(iTStep,iClass).Idx;
                RealDistr = NseData(iTStep).DataTrain(Idx,:);
                PredDistr = GenDistr(iTStep).Data(Idx,:);
                PredMu = mean(PredDistr); 
                PredCov = cov(PredDistr);
                RealCov = cov(RealDistr);
                RealMu = mean(RealDistr);
%                 PredDistrPDF(iClass).Data = mvnpdf(,PredMu,PredCov);
%                 RealDistrPDF(iClass).Data = mvnpdf(,RealMu,RealCov);
                SINDyResults(iTStep,iClass).KLDiv = sqrt(1 - ...
                                                    ((det(RealCov).^(1/4)*det(PredCov).^(1/4))/(det((RealCov+PredCov)/2).^(1/2))).*...
                                                    exp((-1/8)*(RealMu - PredMu)'\((RealCov+PredCov)/2).*(RealMu - PredMu)));
            end
            end
        end
    end
	end
    % END MAIN CODE for Predictive-Optimized Attacks


	% MAIN CODE for Non-Predictive-Optimized Attacks
	if (AttackMethod == "NonPredictiveOptimized")

        for iTStep = 1:NumTimeSteps
        % Can't attack further back then we have information for
        if ((iTStep - TimeStepsInPastToAttack) <= 0)

        % Learn++.NSE gets t = 1 nseResults.net 
        if (iTStep == 1) 
        [NseResults(iTStep).Net,...
         NseResults(iTStep).FMeasure,...
         NseResults(iTStep).GMean,...
         NseResults(iTStep).Precision,...
         NseResults(iTStep).Recall,...
         NseResults(iTStep).Error,...
         NseResults(iTStep).LatestClassifier] = learn_nse_for_attacking(NseResults(iTStep).Net,...
                                                                        NseData(iTStep).DataTrain,...
                                                                        NseData(iTStep).LabelsTrain,...
                                                                        NseData(iTStep).DataTest,...
                                                                        NseData(iTStep).LabelsTest);
        else % Learn++.NSE gets t = t - 1 neseResults
        [NseResults(iTStep).Net,...
         NseResults(iTStep).FMeasure,...
         NseResults(iTStep).GMean,...
         NseResults(iTStep).Precision,...
         NseResults(iTStep).Recall,...
         NseResults(iTStep).Error,...
         NseResults(iTStep).LatestClassifier] = learn_nse_for_attacking(NseResults(iTStep-1).Net,...
                                                                        NseData(iTStep).DataTrain,...
                                                                        NseData(iTStep).LabelsTrain,...
                                                                        NseData(iTStep).DataTest,...
                                                                        NseData(iTStep).LabelsTest);

        end

        % At this point enough time steps have past to start generating attack points based on previous distributions
        else 

        AtkTimeStep = iTStep - TimeStepsInPastToAttack;
        AtkTimeStepData = vertcat(NseData(AtkTimeStep).DataTrain,NseData(AtkTimeStep).DataTest);
        AtkTimeStepLabels = vertcat(NseData(AtkTimeStep).LabelsTrain,NseData(AtkTimeStep).LabelsTest);
        AtkSettings.Boundary = [min(AtkTimeStepData);max(AtkTimeStepData)];
        [AtkData(iTStep).Points,AtkData(iTStep).Labels] = chrisAttacks(AtkTimeStepData,...
                                                                       AtkTimeStepLabels,...
                                                                       AtkSettings);
        [NseResults(iTStep).Net,...
        NseResults(iTStep).Fmeasure,...
        NseResults(iTStep).Gmean,...
        NseResults(iTStep).Precision,...
        NseResults(iTStep).Recall,...
        NseResults(iTStep).Error,...
        NseResults(iTStep).LatestClassifier] = ...
                                learn_nse_for_attacking(NseResults(iTStep-1).Net,...
                                                       [NseData(iTStep).DataTrain;AtkData(iTStep).Points],...
                                                       [NseData(iTStep).LabelsTrain;AtkData(iTStep).Labels'],...
                                                       NseData(iTStep).DataTest,...
                                                       NseData(iTStep).LabelsTest);
        end
    end
    % END MAIN CODE for Non-Predictive-Optimized Attacks

	end

	if (ReturnPlottingInfo)
		PlottingInfo.DataSetInfo = DataSetInfo;
		PlottingInfo.NseData = NseData;
		PlottingInfo.ClassInfo = ClassInfo;
		if (AttackMethod == "PredictiveOptimized")
			PlottingInfo.GenDistr = GenDistr;
			PlottingInfo.SINDyResults = SINDyResults;
			PlottingInfo.AtkData = AtkData;
			PlottingInfo.AtkSettings = AtkSettings;
		end
		if (AttackMethod == "NonPredictiveOptimized")
			PlottingInfo.AtkData = AtkData;
			PlottingInfo.AtkSettings = AtkSettings;
		end
		varargout{1} = PlottingInfo;
	end

end


