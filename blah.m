% @Author: delengowski
% @Date:   2019-05-11 21:29:06
% @Last Modified by:   delengowski
% @Last Modified time: 2019-05-13 20:17:40
DataSet = "X2CDT";

% Load dataset
x = load("Synthetic Datasets\" + DataSet + ".mat");
DataTrain = x.(DataSet).dataTrain;
DataTest = x.(DataSet).dataTest;
LabelsTest = x.(DataSet).labelsTest;
LabelsTrain = x.(DataSet).labelsTrain;

NumTimeSteps = length(DataTrain);
[NumObs,NumDims] = size(DataTrain{1});
NumClasses = numel(unique(LabelsTrain{1}));

DataTrain = cellfun(@(x) mat2cell(x,ones(NumObs,1),NumDims),DataTrain,'UniformOutput',false);

