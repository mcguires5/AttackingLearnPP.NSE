%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%computes the gradient of the rbf kernel wrt x (single point!)
%dQ = rbf(x,y) * Gamma*(2y'-x')
%
% Input parameters :
%   x : a column vector on which the derivate is computed with relevance to
%   y : a matrix reprensenting a set of point to derivate. Each row is an
%   observation and each column represents a feature of data
%   Gamma : Gamma scalar value used with RBF kernel
% Output parameters :
%   dQ : a matrice representing the gradient of the rbf kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dQ = rbf_dQ(x,y,Gamma)


%compute the RBF kernel coeff between x and y
K=rbf(y,x,Gamma);


for i=1:size(y,2) %through features
    
    dQ(:,i) = K.*2.*Gamma.*(y(:,i)-x(i));
    
end

