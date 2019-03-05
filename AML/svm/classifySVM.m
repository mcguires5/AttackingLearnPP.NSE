%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evaluates the performance of a previously trained SVM on a data set
% evaluation is performed using libSVM library
%
% Input parameters :
%   x : matrice containing for each row an observation and where each
%   column represents a feature of the data
%   y : a column vector representing the associated label to data in x
%   model : a precomputed SVM model from libSVM
% Output parameters :
%   y_class : column vector presenting the predicted label of data x using the SVM model
%   score : column vector representing decision values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [y_class score] = classifySVM(x,y,model)

% evaluate the performance of the SVM model on data x
% y_class are the predicted labels while y is the groundtruth
[y_class, ~, score] = svmpredict(y, x, model);

% in case the first label given to libSVM is -1
if(~isempty(model.Label))
   score = score.*model.Label(1);
end