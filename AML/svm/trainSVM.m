%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train a SVM on the given data with given kernel and parameters
%
% Input Parameters :
%   x : a matrice containing for each row the number of observations and
%       where each column represents a feature
%   y : a column vector representing the associated class to data x
%   C : the constraint value defining the cost for the classifier to
%       perform errors
%   Gamma : gamma value used with RBF kernels
% Output Parameters :
%   model : the SVM model built from libSVM
%   alpha : the Lagrangian multiplier ( alpha_i = 0 if point i does not
%           contribute to the separating function, alpha_i = C if the point is
%           between the separating functions and the margins (no matter if it is on
%           the right or wrong side of the separating functions) or 0< alpha_i < C
%           if point i is a Support Vector point
%   b : the bias of the model
%   SV_idx : a column vector containing the index of points considered as
%            Support Vector by libSVM for the computed model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [model alpha b SV_idx] = trainSVM(x,y,C,Gamma)

addpath libsvm

%Gamme parameter is optional
if(nargin<4)
    Gamma=[];
end

% case there is only one class in y 
one_class = (numel(unique(y)) == 1);

%% TRAINING SVM USING libSVM library
% parameters are :
% -s 0|2 : if 2, one class training case else use of C-SVC
% -h 0 : no use of shrinking heuristics
% -t 0|2 : if 0, linear kernel, else RBF kernel
% -c : the cost of errors
% -g : gamma value for the RBF kernel
% -n : nu value for the one class case

% in the case, there is more than 1 class in y
if(one_class == 0)
    %if there is no gamma value => linear kernel
    if( isempty(Gamma) )
        model = fitcsvm(y,x,['-h 0 -t 0 -c ' num2str(C)]);
    else %RBF kernel otherwise
        model = fitcsvm(y,x,['-h 0 -t 2 -g ' num2str(Gamma) ' -c ' num2str(C)]);
    end
else %one class in y
    %if there is no gamma value => linear kernel
    if( isempty(Gamma) )
        model = svmtrain(y,x,['-s 2 -h 0 -t 0 -n ' num2str(C)]);
    else %RBF kernel otherwise
        model = svmtrain(y,x,['-s 2 -h 0 -t 2 -g ' num2str(Gamma) ' -n ' num2str(C)]);
    end
end

% retrieve the Support Vector indexes
[~, SV_idx]=ismember(model.SVs, x,'rows');

% retrieve their alpha value
alpha = zeros(size(x,1),1);
alpha(SV_idx)=abs(model.sv_coef);

% retrieve the bias of the model
b = -model.rho;

