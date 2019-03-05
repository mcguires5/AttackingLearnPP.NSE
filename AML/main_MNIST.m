% Poisoning attacks against SVMs -- Matlab code for running the experiments reported in:
% 
% Battista Biggio, Blaine Nelson, and Pavel Laskov. Poisoning attacks against support vector machines.
% In J. Langford and J. Pineau, editors, 29th Int'€™l Conf. on Machine Learning. Omnipress, 2012.
% 
% http://pralab.diee.unica.it/en/node/729
% http://pralab.diee.unica.it/en/PoisoningAttacks
% 
% Copyright (C) 2013, Battista Biggio, Paul Temple. 
% Dept. of Electrical and Electronic Engineering, University of Cagliari, Italy.
% 
% This code is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This code is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This is the main file of the algorithm presented in the article
%%% Poisoning Attacks Against SVMs published at ICML 2012.
%%% It takes data points from MNIST handwritten digit data set and trains a
%%% SVM classifier on it thantks to libSVM library.
%%% An attack point is then created and added to the training data set.
%%% After evaluating the performance of the new classifier, the attack
%%% point is moved in the feature space in order to find the position
%%% where it has a maximum of influence over the classifier making its
%%% accuracy decreases maximally.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% new context
clear all
close all
clc

% add relative path to the following folders
addpath utils
addpath libsvm
addpath svm
addpath kernels
addpath attack

% select classes to load. In MNIST data set, each class represents a
% number.
% if one wants to change the classes involved in the execution of this
% script, please notice that the classes are declared as string. Also
% notice that the attacking class is the negative one (neg_class) and the class to be
% attacked is the positive one (pos_class).
pos_class = '4';
neg_class = '0';

% load into memory the entire sets given by MNIST data set according to the
% classes declared above. For each set, the positive class appears first
% and then the negative class.
[x_tr, y_tr, x_ts, y_ts] = load_tr_ts_mnist(pos_class,neg_class);

% normalize the pixel value from [0..255]  to [0;1]
x_tr = x_tr ./ 255;
x_ts = x_ts ./ 255;

% definition of the set's cardinalities. The training set will consider
% only 100 points per class, the validation set will contain 500 points and
% the test set 1000 points.
n_tr = 100;
n_vd = 500;
n_ts = 1000;

% the attacking class is the negative one
yc=-1;

% definition of some variables
% - C is the penality coefficient used in soft margin SVMs.
% - Gamma is used only with RBF kernels.
% - num_points is the number of poisoning points to create
% - step is the iteration step
% - lb and ub are feature space's bounds to avoid that attack points deviate
% too much from the training set
C=1; Gamma=[];
num_points=1;
step = 0.05;
lb=0;
ub=1;

% randomly sample the MNIST training set to build training and validation
% sets that the script will use to build the attack
tr_idx = randsample(size(x_tr,1),n_tr+n_vd);

% validation set is built from the n_vd first element randomly sampled
vd_idx = tr_idx(1:n_vd);
x_vd = x_tr(vd_idx,:);
y_vd = y_tr(vd_idx);

% training set is built from the other elements
tr_idx = tr_idx(n_vd+1:end);
x_tr = x_tr(tr_idx,:);
y_tr = y_tr(tr_idx);

% train a SVM from the un-corrupted sets
[model, alpha, b, SV_idx] = trainSVM(x_tr,y_tr,C,Gamma);
% evaluate its performance on the test set
[yclass, ~] = classifySVM(x_ts,y_ts,model);
err_ts = sum(yclass~=y_ts)/numel(y_ts);

% evaluate its performance on the validation test
[yclass, ~] = classifySVM(x_vd,y_vd,model);
err_vd = sum(yclass~=y_vd)/numel(y_vd);

%% run the attack against the SVM trained on uncorrupted data
% the attack is performed as mentionned in the Poisoning Attacks Against
% SVM article

[xc_prime, err, xc] = attackSVM(num_points, step, yc, lb, ub, x_tr, y_tr, x_vd, y_vd, C, Gamma);

% keep into memory the number of errors performed on the validation test
err_vd = [err_vd err];

%computing testing error for each position of the gradient ascent process
for i=1:size(xc,1)
   [model, alpha]= trainSVM([x_tr; xc(i,:)], [y_tr; yc], C, Gamma); 
   [yclass, ~] = classifySVM(x_ts,y_ts,model);
   err_ts(i+1) = sum(yclass~=y_ts)/numel(y_ts);
   
end

% save all variables into a file
% filename = ['exp_single_attack/tmp_' pos_class '_vs_' neg_class '.mat'];
% save(filename)

%% visualisation of errors and of the final image

% load the file
% load filename

addpath exportfig

% parameters of the image
fontsize=10;
w=20;
h=10;

[fig1, opts1]=createfig(fontsize,w,h);

% the first sub-image will display the digit image at the begining of the
% attack (without displacement of the point)
subplot(1,3,1)
display_character(xc(1,:));
title(['Before attack (' pos_class ' vs ' neg_class ')'])
axis square

% the second sub-image will display the digit image at the end of the
% attack (after the gradient ascent process is over)
subplot(1,3,2)
display_character(xc_prime);
title(['After attack (' pos_class ' vs ' neg_class ')'])
axis square

% the last sub-image shows the validation and test classification error
% perform by the SVM trained on the initial training set plus the attack
% point
subplot(1,3,3)
plot(err_vd,'r')
grid on
hold on
plot(err_ts,'k--')
legend('validation error', 'testing error')
axis square
xlabel('number of iterations')
title('classification error')
legend('location','northwest')

axis([0 400 0 0.4])

% save the plot
exportfig(fig1,['fig/' pos_class '_vs_' neg_class '.eps'],opts1);

