function [] = plotResultsAdvLearnSINDy(GenDistr,AtkData,NseResults,timeStep,svmNumber,svmMatch,NseData,plotGenData)
    figure
    hold on
    gscatter(NseData(timeStep).DataTrain(:,1),NseData(timeStep).DataTrain(:,2),NseData(timeStep).LabelsTrain,'rb','.')
    if plotGenData == true
        gscatter(GenDistr(timeStep).Data(:,1),GenDistr(timeStep).Data(:,2),GenDistr(timeStep).Labels,'rb','*')
    end
    if svmMatch == true
        mdl = NseResults(timeStep).Net.classifiers{1, timeStep}.classifier;
    else
        mdl = NseResults(timeStep).Net.classifiers{1, svmNumber}.classifier;
    end
    maxBounds = max(NseData(timeStep).DataTrain);
    minBounds = min(NseData(timeStep).DataTrain);
    X1 = linspace(minBounds(1),maxBounds(1));
    X2 = linspace(minBounds(2),maxBounds(2));
    [x1 x2] = meshgrid(X1,X2);
    for i = 1:size(x1,2)
        myGrid = [x1(:,i) x2(:,i)];
        gridScores(:,i) = predict(mdl, myGrid);
    end
    [one,two] = contour(x1, x2, gridScores,1,'LineWidth',3);
    plot(mdl.SupportVectors(:,1),mdl.SupportVectors(:,2),'^','MarkerSize',10,'Color',[0.4660 0.6740 0.1880],'LineWidth',1)
    plot(AtkData(timeStep).Points(:,1),AtkData(timeStep).Points(:,2),...
        'x','Color',[0.8500 0.3250 0.0980],'LineWidth',3)
    hold off
    title("Time Step = " + timeStep +", Error = "+NseResults(timeStep).Error*100+"%")
    if plotGenData == true
            legend('Class 1 Train','Class 2 Train','Gen C1','Gen C2','boundary','support vectors','Attack Points','Location','eastoutside')
    else
        legend('Class 1 Train','Class 2 Train','boundary','support vectors','Attack Points','Location','eastoutside')
    end
end
