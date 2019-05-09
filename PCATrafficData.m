 [coeff,score] = pca(trafficCars.data);
 Comp_PCA1 = score(:,1:10);
 
 for i = 1:length(Comp_PCA1)/20
    dataTrain{i} = Comp_PCA1(((i-1)*20)+1:(i*20)-3,:);
    labelsTrain{i} = trafficCars.labels(((i-1)*20)+1:(i*20)-3,:);
    dataTest{i} = Comp_PCA1(((i-1)*20)+18:i*20,:);
    labelsTest{i} = trafficCars.labels(((i-1)*20)+18:i*20,:);
 end
 dataTrain = cell(dataTrain);
 trafficCar.dataTrain = dataTrain;
 dataTest = cell(dataTest);
 trafficCar.dataTest = dataTest;
 labelsTrain = cell(labelsTrain);
 trafficCar.labelsTrain = labelsTrain;
 labelsTest = cell(labelsTest);
 trafficCar.labelsTest = labelsTest;
 %trafficCars = struct('dataTrain',dataTrain,'labelsTrain',labelsTrain,'dataTest',dataTest,'labelsTest',labelsTest);
 save('trafficCars.mat','trafficCar')
