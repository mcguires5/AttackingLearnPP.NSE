%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% create the background of the figure in order to visualizeand search for
% the direction which will lead to a maximum of error performed
% by the SVM classifier
%
% Input parameters :
%   x : matrices representing the training data set (plus the attack point)
%       where a row represents an observation and a column a feature of the
%       data
%   y : column vector representing the associated labels
%   x_tst : matrice of evaluation data set where each row represents an
%           observation and a column is a feature of the data
%   y_tst : column vector representing the associated labels
%   yc : integer representing the label of the attack point
%   C : scalar representing the cost incurred by the classifier by making
%       errors
%   Gamma : scalar representing the gamma value used with RBF kernel
%   grid_coords : a row vector containing the dimension of the matrice
%                 representing the feature space
%   grid_size : the size of each patch of the feature space
% Output parameters :
%   x_testa1 : matrice containing the coordinates of the grid on the
%              positive side
%   x_testa2 : matrice containing the coordinates of the grid on the
%              negative side
%   err_test : matrice containing the test errors performed by the
%              classifier for each part of the grid 
%   err_sum_gi : matrice containing the sum of g_i performed by the classifier
%                for each part of the grid 
%   err_xi : matrice containing the sum of negative g_i (error points) 
%            performed by the classifier for each part of the grid 
%   err_logloss : matrice containing the error of the hinge loss function
%                 for each part of the grid
%   err_span : estimate of the number of errors made by leave-one-out
%              procedure on the SVM classifier
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xtesta1 xtesta2 err_test err_sum_gi err_xi err_logloss err_span] = grid_search_optimal_attack(x,y,x_tst,y_tst,yc,C,Gamma,grid_coords,grid_size)


%----------------------------
% create test set
%----------------------------
% genererate a grid from grid_coords(1) to grid_coords(2) on x-axis
% and from grid_coords(3) to grid_coords(4) on y-axis
% the grid is divided in grid_size elements on each axis
[xtesta1,xtesta2]=meshgrid( ...
     linspace(grid_coords(1), grid_coords(2), grid_size), ...
     linspace(grid_coords(3), grid_coords(4), grid_size) );
 [na,nb]=size(xtesta1);
 % matrices to column vectors
 xtest1=reshape(xtesta1,1,na*nb);
 xtest2=reshape(xtesta2,1,na*nb);
 xtest=[xtest1;xtest2]';
 
 % column vectors representing different errors
 err_test = zeros(1,na*nb);
 err_sum_gi = zeros(1,na*nb);
 err_xi = zeros(1,na*nb);
 err_span = zeros(1,na*nb);
 err_logloss = zeros(1,na*nb);
 
 % case of one class
 one_class = (numel(unique(y)) == 1);
 
 for i=1:size(xtest,1)
      
    %estimating the test error
    %train an SVM by adding one point from xtest at a time
    [model alpha b] = trainSVM([x; xtest(i,:)],[y; yc],C,Gamma);

    
    %evaluate the performance of the SVM model on x_tst
    [yclass score] = classifySVM(x_tst,y_tst,model);
    gi=y_tst.*score -1;
    if(one_class)
        gi=gi+1;
    end
    xi = max(0,-gi);
    
    %update errors
    err_test(i) = sum(yclass~=y_tst)/size(x_tst,1);
    err_sum_gi(i) = -sum(gi);
    err_xi(i)=sum(xi);
    err_span(i)=fast_span_estimate(lin([x; xtest(i,:)],[x; xtest(i,:)]),[y; yc],alpha,b,C);
    
    err_logloss(i) = sum(log(1+exp(-gi-1)));
    
 end

 %from column vectors to matrices
err_test = reshape(err_test,na,nb);
err_sum_gi = reshape(err_sum_gi,na,nb);
err_xi = reshape(err_xi,na,nb);
err_span = reshape(err_span,na,nb);
err_logloss = reshape(err_logloss,na,nb);
