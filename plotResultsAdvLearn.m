function [] = plotResultsAdvLearn(atkData,nseResults,timeStep,svmNumber,svmMatch,nseData,timeStepsInPastToAttack,plotPrevData)
    figure
    hold on
    gscatter(nseData(timeStep).dataTrain(:,1),nseData(timeStep).dataTrain(:,2),nseData(timeStep).labelsTrain,'rb','.')
    if plotPrevData == true
        atkTimeStep = timeStep - timeStepsInPastToAttack;
        atkTimeStepData = vertcat(nseData(atkTimeStep).dataTrain,nseData(atkTimeStep).dataTest);
        atkTimeStepLabels = vertcat(nseData(atkTimeStep).labelsTrain,nseData(atkTimeStep).labelsTest);
        gscatter(atkTimeStepData(:,1),atkTimeStepData(:,2),atkTimeStepLabels(:,1),'rb','*')
    end
    if svmMatch == true
        mdl = nseResults(timeStep).net.classifiers{1, timeStep}.classifier;
    else
        mdl = nseResults(timeStep).net.classifiers{1, svmNumber}.classifier;
    end
    maxBounds = max(nseData(timeStep).dataTrain);
    minBounds = min(nseData(timeStep).dataTrain);
    X1 = linspace(minBounds(1),maxBounds(1));
    X2 = linspace(minBounds(2),maxBounds(2));
    [x1 x2] = meshgrid(X1,X2);
    for i = 1:size(x1,2)
        myGrid = [x1(:,i) x2(:,i)];
        gridScores(:,i) = predict(mdl, myGrid);
    end
    [one,two] = contour(x1, x2, gridScores,1,'LineWidth',3);
    plot(mdl.SupportVectors(:,1),mdl.SupportVectors(:,2),'^','MarkerSize',10,'Color',[0.4660 0.6740 0.1880],'LineWidth',1)
    plot(atkData(timeStep).points(:,1),atkData(timeStep).points(:,2),...
        'x','Color',[0.8500 0.3250 0.0980],'LineWidth',3)
    hold off
    title("Time Step = " + timeStep +", Error = "+nseResults(timeStep).errs_nse*100+"%")
    if plotPrevData 
        legend('Class 1 Train','Class 2 Train','Prev C1','Prev C2','boundary','support vectors','Attack Points','Location','eastoutside')
    else
   legend('Class 1 Train','Class 2 Train','boundary','support vectors','Attack Points','Location','eastoutside')
    end
end