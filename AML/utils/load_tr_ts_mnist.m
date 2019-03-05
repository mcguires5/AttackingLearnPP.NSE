%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a function to load the classes of MNIST data set defined by the user in
% the main script performing an attack against SVM as described in the
% article Poisoning Attacks Against SVMs published in ICML 2012.
%
% Input parameters :
%   pos_class : a string representing the positive class defined by a digit
%   neg_class : a string representing the negative class defined by a digit
%
% Output parameters :
%   x_tr : a matrix containing the training sets of both positive and
%   negative classes. Positive class is stored first.
%   Those training sets are only loaded from the MNIST
%   data sets, no preprocessing has been performed yet
%   y_tr : a column vector containing the classes of each element of the
%   training set. The number of rows in y_tr and x_tr are then equal.
%   Positive class appears first with the label '1'. The negative class
%   comes after with the label '-1'.
%   x_ts : a matrix containing the test sets of both positive and
%   negative classes. Positive class is stored first.
%   Those training sets are only loaded from the MNIST
%   data sets, no preprocessing has been performed yet
%   y_ts : a column vector containing the classes of each element of the
%   test set. The number of rows in y_tr and x_tr are then equal.
%   Positive class appears first with the label '1'. The negative class
%   comes after with the label '-1'.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x_tr y_tr x_ts y_ts] = load_tr_ts_mnist(pos_class,neg_class)

% load all the data from MNIST data set
D = load('dataset/mnist_all.mat');

% keep only the training and test set of the class defined by the user
% (input parameters)
pos_tr = eval(['D.train' pos_class]);
neg_tr = eval(['D.train' neg_class]);

pos_ts = eval(['D.test' pos_class]);
neg_ts = eval(['D.test' neg_class]);

% concatenate the sets of the two classes to build the training and test
% set. And finally, create associated labels
x_tr = double([pos_tr; neg_tr]);
y_tr = double([ones(size(pos_tr,1),1); -ones(size(neg_tr,1),1)]);

x_ts = double([pos_ts; neg_ts]);
y_ts = double([ones(size(pos_ts,1),1); -ones(size(neg_ts,1),1)]);

return;
