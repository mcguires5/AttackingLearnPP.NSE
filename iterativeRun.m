pathsForCode();
close all
clc
DataSet = "X2CDT_FastMovingx8";
Kernel = "Linear";
lambda = 8e-3;
polyOrder = 1;
useSine = 0;
sineMultiplier = 10;
useExp = 0;
expMultiplier = 10;
maxSteps = 150; %250
stepSize = 10; %5
degree = 3;
coef0 = 1;
c = 1; % Class to attack
numAtkPts = 40;
timeStepsInPastToAttack = 4;
f = waitbar(0,'Running baseline Learn++.NSE');
[nseData_Baseline,nseResults_Baseline,...
	   ~,~,~,~] = ...
	   LearnPlusPLusNSE_Baseline(DataSet,... % General setting
                                     Kernel); 

numRunTimes = 100;
Errors = cell(numRunTimes,2);
Errors(:) = {zeros(1,40)};

profile on
for iRT = 1:numRunTimes
waitbar(iRT/numRunTimes,f,"Performing attack iteration " + iRT + "/" + numRunTimes + "(Predictive Attack)");    

[NseResultsOptAtks,PlottingInfo] = ...
AttackingLearnPlusPlusDotNSE(DataSet,"PredictiveOptimized",Kernel,...
                             "Lambda",lambda,"PolyOrder",polyOrder,...
                             "MaxSteps",maxSteps,"StepSize",stepSize,...
                             "Degree",degree,"Coef0",coef0,...
                             "ClassToAttack",c,"NumAttackPoints",numAtkPts,...
                             "TimeStepsInPastToAttack",timeStepsInPastToAttack,...
                             "ReturnPlottingInfo",1);

%Errors{iRT,1} = [NseResultsOptAtks.Error];
ErrorAdvLearnSINDy = [NseResultsOptAtks.Error];
waitbar(iRT/numRunTimes,f,"Performing attack iteration " + iRT + "/" + numRunTimes + "(Non Predictive Attack)");

[NseResultsAtks] = ...
AttackingLearnPlusPlusDotNSE(DataSet,"NonPredictiveOptimized",Kernel,...
                             "MaxSteps",maxSteps,"StepSize",stepSize,...
                             "Degree",degree,"Coef0",coef0,...
                             "ClassToAttack",c,"NumAttackPoints",numAtkPts,...
                             "TimeStepsInPastToAttack",timeStepsInPastToAttack);

%Errors{iRT,2} = [NseResultsAtks.Error];
ErrorAdvLearn  = [NseResultsAtks.Error];

blah1 = readmatrix("AdvLearnSINDy.txt");
blah1 = [blah1;ErrorAdvLearnSINDy];
writematrix(blah1,"AdvLearnSINDy.txt");

blah2 = readmatrix("AdvLearn.txt");
blah2 = [blah2;ErrorAdvLearn];
writematrix(blah2,"AdvLearn.txt");

end
profile viewer

AdvLearnError = mean(cell2mat(vertcat(Errors(:,2))));
AdvLearnSINDyError = mean(cell2mat(vertcat(Errors(:,1))));
BaselineError = [nseResults_Baseline(1:40).errs_nse];

Error(1,:) = BaselineError;
Error(2,:) = AdvLearnSINDyError(1:end);
Error(3,:) = AdvLearnError(1:end);
% Error(2,:) = cell2mat(vertcat(Errors(:,1)));
% Error(3,:) = cell2mat(vertcat(Errors(:,2)));
figure
plot((1:40)',Error'*100,'LineWidth',3)
xlabel('Timestep')
ylabel('Error')
legend('Baseline','AdvLearnSINDy','AdvLearn','Location','SouthEast')