function [] = plotResults(classInfo,genDistr,atkData,nseResults,timeStep)
    classColors = ["r" "b"];
    figure
    hold on
    for iClass = 1:size(classInfo,2)
    plot(genDistr(timeStep).data(genDistr(timeStep).labels == iClass,1),...
         genDistr(timeStep).data(genDistr(timeStep).labels == iClass,2),...
         classColors(iClass)+".")
     plot(classInfo(timeStep,iClass).data(:,1),classInfo(timeStep,iClass).data(:,2),...
          classColors(iClass)+"*")
    end
    plot(atkData(timeStep).points(:,1),atkData(timeStep).points(:,2),...
        'x','Color',[0.8500 0.3250 0.0980],'LineWidth',3)
    maxBounds = max(genDistr(timeStep).data);
    minBounds = min(genDistr(timeStep).data);
        % need to construct boundary for n dimensional dataset
    mdl = nseResults(timeStep).net.classifiers{1, end}.classifier;
    f = @(x) -(x*mdl.Beta(1) + mdl.Bias)/mdl.Beta(2);
    x = linspace(minBounds(1),maxBounds(1));
    plot(x,f(x),'g--','LineWidth',2)
    plot(mdl.SupportVectors(:,1),mdl.SupportVectors(:,2),'kO','MarkerSize',3)
    hold off
    legend('Gen C1','Real Tr C1','Gen C2','Real Tr C2','Atk Points','Boundary','Support Vectors','Location','SouthEast')
end