%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the kernel function of the linear kernel K(xi,xj)=xi*xj'
%
% Input parameters :
%   x_ts : Matrix containing the validation set. Rows are observations and
%          each column is a feature of the data
%   x_tr : Matrix containing the training set. Rows are observations and
%          each column is a feature of the data
% Output parameters :
%   K : the kernel function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function K = lin(x_ts,x_tr)

K=x_ts*x_tr';