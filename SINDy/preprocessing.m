% @Author: Delengowski_Mobile
% @Date:   2019-04-25 12:38:40
% @Last Modified by:   Delengowski_Mobile
% @Last Modified time: 2019-04-25 13:12:58

function [ ] = attacksPreprocessing(classInfo,numClasses,numDims,numObs,numTimeSteps,vargin)

	p = inputParser;

	addRequired(p,'numObs',@isnumeric);
	addRequired(p,'numDims',@isnumeric);
	addRequired(p,'numTimeSteps',@isnumeric);
	addRequired(p,'numClasses',@isnumeric);
	addRequired(p,'classInfo',@isstruct);

	% Attack Types
	addOptional(p,'SINDy',false);
	addOptional(p,'OptimizeAttacks',false);



	% ADvlearn Settings
	addOptional(p,'kernel','Linear');
	addOptional(p,'stepSize',5);
	addOptional(p,'maxSteps',250);
	addOptional(p,'degree',3);
	addOptional(p,'classToAttack',1);
	addOptional(p,'ceof0',1);
	addOptional(p,'gamma','auto')
	addOptional(p,'numAtkPts',);

	% SINDy Settings
	addOptional(p,'lambda',8e-3);
	addOptional(p,'polyOrder',1);
	addOptional(p,'useSine',0);
	addOptional(p,'sineMultiplier',10);
	addOptional(p,'useExp',0);
	addOptional(p,'expMultiplier',10);


	if "OptimizeAttacks" == true

		atkSet.step_size = stepSize;
		atkSet.timeToAttack = 3;
		atkSet.max_steps = maxSteps;
		atkSet.c = classToAttack;
		atkSet.kernel = lower(kernel);
		atkSet.degree = degree;
		atkSet.coef0 = coef0;
		atkSet.gamma = 'auto';
		atkSet.numAtkPts = numAtkPts;

	end

	if "SINDy" == true

	% SINDyData is structure array (timestep,class), stores the means and sigma to feed into SINDy at each time step
	SINDyData = repmat(struct("mu",0,"sigma",0),numTimeSteps,numClasses);
	for iClass = 1:numClasses
	    for iTStep = 2:numTimeSteps
	        SINDyData(iTStep,iClass).mu = vertcat(classInfo(1:iTStep,iClass).mu);
	        SINDyData(iTStep,iClass).sigma = classInfo(iTStep-1,iClass).sigma;
	    end
	end
	[SINDyData.latestReset] = deal(1);
	% SINDyResults is a structure array (timestep,class), stores the predicted mean and table of learned functions 
	SINDyResults = repmat(struct("prediction",0,"learnedFunctions",0),numTimeSteps,numClasses);
	% Create a SINDy models. One per class
	for iClass = numClasses:-1:1
	    SINDyArray(1,numClasses) = SINDy(); % SINDy Object ... type help SINDy for list of what each parameter does
	end
	[SINDyArray.lambda] = deal(lambda);
	[SINDyArray.polyOrder] = deal(polyOrder);
	[SINDyArray.useSine] = deal(useSine);
	[SINDyArray.sineMultiplier] = deal(sineMultiplier);
	[SINDyArray.useExp] = deal(useExp);
	[SINDyArray.expMultiplier] = deal(expMultiplier);
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

	if ("SINDy" == true && "OptimizeAttacks" == true)

	% variables to hold attack points and generated data sets
	genDistr = repmat(struct("data",zeros(numObs,numDims),"labels",zeros(numObs,1)),1,numTimeSteps);
	atkData = repmat(struct("points",zeros(atkSet.numAtkPts,numDims),...
	                           "labels",zeros(atkSet.numAtkPts,1)),1,numTimeSteps);
	[genDistr(1:atkSet.timeToAttack-1).data] = deal(0);
	[genDistr(1:atkSet.timeToAttack-1).labels] = deal(0);
	[atkData(1:atkSet.timeToAttack-1).points] = deal(0);
	[atkData(1:atkSet.timeToAttack-1).labels] = deal(0);
	thereIsTimeToAttack = true;

	end

	if ("SINDy" == false && "OptimizeAttacks" == false)

	% variables to hold attack points and generated data sets
	atkData = repmat(struct("points",zeros(atkSet.numAtkPts,numDims),...
	                           "labels",zeros(atkSet.numAtkPts,1)),1,numTimeSteps);
	[atkData(timeStepsInPastToAttack:end).points] = deal(0);
	[atkData(timeStepsInPastToAttack:end).labels] = deal(0);

	end

end