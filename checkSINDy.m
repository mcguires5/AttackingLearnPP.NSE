function [drift] = checkSINDy(SINDyArray,SINDyData,StdevThresh,iTStep,iClass)
    train = SINDyArray(iTStep-1,iClass).data;
    curData = SINDyData(iTStep,iClass).mu;    
    SINDY = SINDyArray(iTStep-1,iClass).model;
    [drift, ~] = detectDrift(train, curData, SINDY, StdevThresh);
end