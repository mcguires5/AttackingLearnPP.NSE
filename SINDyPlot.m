function [] = SINDyPlot(class,timeStep,SINDyArray,SINDyData)
    figure
    hold on
    scatter(SINDyArray(timeStep-1,class).model(1:end-1,1),SINDyArray(timeStep-1,class).model(1:end-1,2),40,[0 0.4470 0.7410],'filled')
    scatter(SINDyArray(timeStep-1,class).model(end,1),SINDyArray(timeStep-1,class).model(end,2),40,[0 0.4470 0.7410],'filled','MarkerEdgeColor',[0.9290 0.6940 0.1250],'LineWidth',2)
    scatter(SINDyData(timeStep,class).mu(1:end-1,1),SINDyData(timeStep,class).mu(1:end-1,2),40,[0.6350 0.0780 0.1840],'filled')
    scatter(SINDyData(timeStep,class).mu(end,1),SINDyData(timeStep,class).mu(end,2),40,[0.6350 0.0780 0.1840],'filled','MarkerEdgeColor',[0.9290 0.6940 0.1250],'LineWidth',2)
    legend("SINDy","Relevent SINDy Pred","True","True Current Time Step","Location","bestoutside")
    SINDyArray(timeStep,2).learnedFunctions
    hold off
end
