pathsForCode();
dataset = "X2CDT";
kernel = "Linear";
lambda = 8e-3;
polyOrder = 1;
useSine = 0;
sineMultiplier = 10;
useExp = 0;
expMultiplier = 10;
maxSteps = 250;
stepSize = 5;
degree = 3;
coef0 = 1;
c = 2; % Class to attack
numAtkPts = 40;

[nseData_Baseline,nseResults_Baseline,...
	   ~,~,~,~] = ...
	   LearnPlusPLusNSE_Baseline(dataset,... % General setting
                                     kernel); 

numRunTimes = 5;
Errors = cell(numRunTimes,2);
Errors(:) = {zeros(1,21)};
profile on                     
for iRT = 1:numRunTimes
    
[classInfo_AdvLearnSINDy,...
nseData_AdvLearnSINDy,nseResults_AdvLearnSINDy,...
numClasses,numDims,numObs,numTimesSteps,...
SINDyData_AdvLearnSINDy,SINDyResults_AdvLearnSINDy,...
atkData_AdvLearnSINDy,genDistr_AdvLearnSINDy] = ...
...
attackingLearnPlusPlus_chrisAttack_SINDy(dataset,... % General setting
                                        kernel,... % Learn++.NSE & AttackSVM setting
                                        lambda,polyOrder,... % SINDy Setting
                                        useSine,sineMultiplier,... % SINDy Setting
                                        useExp,expMultiplier,... % SINDy Setting
                                        maxSteps,stepSize,... % AttackSVM setting
                                        degree,coef0,... % AttackSVM setting
                                        numAtkPts,c); % AttackSVM setting
Errors{iRT,1} = nseResults_AdvLearnSINDy.errs_nse;

timeStepsInPastToAttack = 4;
[nseData_AdvLearn,nseResults_AdvLearn,...
~,~,~,~,...
atkData_AdvLearn] = attackingLearnPlusPlus_AdvLearn(dataset,... % General setting
                                           kernel,... % Learn++.NSE & AttackSVM setting
    		                           maxSteps,stepSize,... % AttackSVM setting
                		           degree,coef0,... % AttackSVM setting
                		           numAtkPts,... % AttackSVM setting
                                           timeStepsInPastToAttack,c); % AttackSVM setting
Errors{iRT,2} = nseResults_AdvLearn.errs_nse;
end

profile viewer