function [drift] = checkSINDy(SINDyResults,SINDyData,StdevThresh,iTStep,iClass)
    train = SINDyResults(1,iClass).data;
    curData = SINDyData(iTStep,iClass).mu;    
    model = SINDyResults(1,iClass).model;
    [drift, ~] = detectDrift(train, curData, model, StdevThresh);
end