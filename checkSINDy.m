function [drift] = checkSINDy(SINDyResults,SINDyData,StdevThresh,iTStep,iClass)
    model = SINDyResults(iTStep-1,iClass).Prediction;
    curData = SINDyData(iTStep,iClass).Mu;    
    train = SINDyResults(1,iClass).Prediction; % bullshit line, not currently used.
    [drift, ~] = detectDrift(train, curData, model, StdevThresh);
end