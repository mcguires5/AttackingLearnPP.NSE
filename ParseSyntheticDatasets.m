files = dir('Nonstationary Datasets(Processed)/*.mat');
for file = files'
    x = load(file.name);
    disp(file.name)
    obj = x.(file.name(1:end-4));
    labels = obj.labels;
    data = obj.data;
    if mod(length(data),obj.drift) == 0   
        data2 = reshape(data, obj.drift, size(data,2),length(data)/obj.drift);
        labels2 = reshape(labels, obj.drift, 1,length(labels)/obj.drift);
        for iDepth = 1:length(data)/obj.drift
            TrainDataCell{iDepth} = data2(1:length(data2)/2,:,iDepth)';
            TrainlabelsCell{iDepth} = labels2(1:length(data2)/2,:,iDepth)';
            TestDataCell{iDepth} = data2(length(data2)/2+1:end,:,iDepth)';
            TestlabelsCell{iDepth} = labels2(length(data2)/2+1:end,:,iDepth)';
        end
        newstruct = 'train_data';
        [obj.(newstruct)] = TrainDataCell;
        newstruct = 'train_labels';
        [obj.(newstruct)] = TrainlabelsCell;
        newstruct = 'test_labels';
        [obj.(newstruct)] = TestlabelsCell;
        newstruct = 'test_data';
        [obj.(newstruct)] = TestDataCell;
        eval([file.name(1:end-4),' = obj'])
        % Do some stuff
        save("Synthetic Datasets\" + file.name,file.name(1:end-4));
    end
end