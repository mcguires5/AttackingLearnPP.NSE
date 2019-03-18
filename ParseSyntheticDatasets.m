files = dir('Nonstationary Datasets(Processed)/*.mat');
for file = files'
    clearvars -except file files
    x = load(file.name);
    disp(file.name)
    obj = x.(file.name(1:end-4));
    labels = obj.labels;
    classLabels = unique(labels);
    data = obj.data;
    idx = [obj.trainInd(1,:);obj.testInd];
    for iTStep = 1:obj.numBatches+1
       dataStep = data(idx(iTStep,1):idx(iTStep,2),:);
       labelStep = labels(idx(iTStep,1):idx(iTStep,2),:);
       for iClass = classLabels'
          classData = dataStep(labelStep == iClass,:);
          linearIdx = 1:size(classData,1);
          if iClass == 1
              [trainSet,trainSetIdx] = datasample(classData,floor(size(classData,1)/2),'Replace',false);
              dataTrain{iTStep} = trainSet;
              labelsTrain{iTStep} = repmat(iClass,floor(size(classData,1)/2),1);
              labelsTest{iTStep} = repmat(iClass,floor(size(classData,1)/2),1);
              dataTest{iTStep} = setdiff(classData,trainSet,'rows');
          elseif iClass > 1
           trainSet = datasample(classData,floor(size(classData,1)/2),'Replace',false);
           dataTrain{iTStep} = [dataTrain{iTStep};trainSet];
           labelsTrain{iTStep} = [labelsTrain{iTStep};repmat(iClass,floor(size(classData,1)/2),1)];
           labelsTest{iTStep} = [labelsTest{iTStep};repmat(iClass,floor(size(classData,1)/2),1)];
           dataTest{iTStep} = [dataTest{iTStep};setdiff(classData,trainSet,'rows')];
           end
       end
    end
   blah.dataTrain = dataTrain;
   blah.labelsTrain = labelsTrain;
   blah.labelsTest = labelsTest;
   blah.dataTest = dataTest;
   eval([file.name(1:end-4),' = blah;']);
    % Do some stuff
    save("Synthetic Datasets\" + file.name,file.name(1:end-4));
end