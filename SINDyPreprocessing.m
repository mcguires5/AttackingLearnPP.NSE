function [SINDyData,SINDyArray,SINDyResults] = SINDyPreprocessing(classInfo,...
	                                                          numClasses,numTimeSteps,...
	                                                          lambda,polyOrder,...
	                                                          useSine,sineMultiplier,...
	                                                          useExp,expMultiplier)
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