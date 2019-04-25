function [atkSet,atkData,...
	  thereIsTimeToAttack,...
	  genDistr] = chrisAttacks_With_SINDy_Preprocessing(kernel,...
	                                                    maxSteps,stepSize,....
	                                                    degree,coef0,...
	                                                    numAtkPts,...
                                                        numObs,numDims,numTimeSteps,...
                                                        c)
	atkSet.step_size = stepSize;
	atkSet.timeToAttack = 3;
	atkSet.max_steps = maxSteps;
	atkSet.c = c;
	atkSet.kernel = lower(kernel);
	atkSet.degree = degree;
	atkSet.coef0 = coef0;
	atkSet.gamma = 'auto';
	atkSet.numAtkPts = numAtkPts;
	% Mesh and step is for creating boundary of attack
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
