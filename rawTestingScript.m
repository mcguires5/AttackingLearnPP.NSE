pathsForCode();
dataset = "X2CDT_FastMoving";
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
c = 1; % Class to attack
numAtkPts = 40;

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
                                        numAtkPts,c,... % AttackSVM setting
                                        StdevThresh); 