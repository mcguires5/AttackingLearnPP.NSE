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
%%% It creates data points in a 2D space and trains a SVM classifier on it
%%% thanks to LibSVM library.
%%% An attack point is then created and added to the training data set.
%%% After evaluating the performance of the new classifier, the attack
%%% point is moved in the 2D feature space in order to find the position
%%% where it has a maximum of influence over the classifier making its
%%% accuracy decreases maximally.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear context
clc
clear all
close all

addpath libsvm
addpath svm
addpath kernels
addpath utils
addpath attack

%parameter given in the article + parameter of the boxconstraint
C=1;
Gamma=[]; %linear kernel
% Gamma = 0.5; %uncomment for RBF kernel

nu=[];%0.1;

grid_size=10;
%grid_coords = [-2 2 -2 2];
grid_coords = [-5 5 -5 5];
%grid_coords = [-10 10 -10 10];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ATTACK PARAMETERS
%

%step size given in the article
step = 0.15;
%step = 0.01;
%number of attack points and coordinate + label
num_points = 1;
yc = 1;
xc = [2 0];

%box constraints on xc
%(lower and upper values for each feature)

lb = -4;
ub = 4;
%lb = -1.5;
%ub = 1.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% DATASETS
%

 %TR and TS samples/class
 n=25; m=500;

 %Gaussian DATA
 mu=1.5; sigma = 0.6; num_feat=2;

 %training data set
 [x, y] = load_gaussian_data(mu,sigma,n,num_feat);

 x = [x; xc]; %add the attack point
 y = [y; yc];

 %evaluation set
 [x_tst, y_tst] = load_gaussian_data(mu,sigma,m,num_feat);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% in case of one class SVM
if(~isempty(nu))
    C=nu; %we set this for convenience in passing the parameter to our functions
    x = x(y==+1,:);
    y = y(y==+1);
end


%% Generating grid for the objective function (to display the colored background)

%trains SVM and evaluates its performance
[model, alpha, b, SV_idx] = trainSVM(x,y,C,Gamma);
[~, score]=classifySVM(x_tst,y_tst,model);

[x1, x2, err_tst, err_sum_gi, err_xi, err_logloss, err_span] = grid_search_optimal_attack(x,y,x_tst,y_tst,yc,C,Gamma,grid_coords,grid_size);


%% Gradient-based attack

% attack point's position
init_points = x(end,:);

%if multiple points are initialized, only the best attack is returned by attackSVM
%init_points = x(alpha>0 & alpha<C & y==yc,:); %support vectors instances

[xc_last err xc alpha_c xi_obj] = attackSVM(num_points, step, yc, lb, ub, x, y, x_tst, y_tst, C, Gamma,init_points);

[yclass gc]=classifySVM(xc,yc*ones(size(xc,1),1),model);

%g of the attack point
gc = yc*gc-1;



%% Plots

%fix for hinge loss
xi_obj = 1/(2*m)*xi_obj;
err_xi = 1/(2*m)*err_xi;

addpath exportfig

fontsize=10;
w=10;
h=10;

[fig1 opts1]=createfig(fontsize,w,h);
contourf(x1,x2,err_tst,50);
caxis([0 max(max(err_tst))])
%caxis([0.02 max(max(err_tst))])
colorbar
shading flat;
axis square
title('classification error')
hold on;
plotSVM(x,y,model,grid_coords);
%display box constraints
boxc = lb:0.1:ub;
plot(boxc,ub*ones(1,numel(boxc)),'k--');
plot(boxc,lb*ones(1,numel(boxc)),'k--');
plot(ub*ones(1,numel(boxc)),boxc,'k--');
plot(lb*ones(1,numel(boxc)),boxc,'k--');
plot(xc(:,1),xc(:,2),'k.-','MarkerSize',9);
axis(grid_coords)


[fig2 opts2]=createfig(fontsize,w,h);contourf(x1,x2,err_xi,50);
caxis([0 max(max(err_xi))])
%caxis([0.11 max(max(err_xi))])
colorbar
shading flat;
axis square
title('mean \Sigma_i \xi_i (hinge loss)')
hold on;
plotSVM(x,y,model,grid_coords);
%display box constraints
boxc = lb:0.1:ub;
plot(boxc,ub*ones(1,numel(boxc)),'k--');
plot(boxc,lb*ones(1,numel(boxc)),'k--');
plot(ub*ones(1,numel(boxc)),boxc,'k--');
plot(lb*ones(1,numel(boxc)),boxc,'k--');
plot(xc(:,1),xc(:,2),'k.-','MarkerSize',9);
axis(grid_coords)

% figure
% subplot(1,2,1)
% plot(err,'b.-')
% title('classification error')
% grid on
% 
% subplot(1,2,2)
% plot(xi_obj,'r.-');
% title('average \Sigma_i \xi_i')
% grid on
% 
% 
% figure
% subplot(1,2,1)
% plot(alpha_c,'r.-')
% title('\alpha_c')
% grid on
% 
% subplot(1,2,2)
% plot(gc,'b.-')
% title('g_c')
% grid on

exportfig(fig1,'fig/fig1.eps',opts1);
exportfig(fig2,'fig/fig2.eps',opts2);

%Converts all eps files within the fig/ folder to pdf.
eps2pdf('fig'); %Please set location to your epstopdf converter inside the eps2pdf.m file, if this does not work.



