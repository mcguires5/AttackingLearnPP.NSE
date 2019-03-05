function [loo span] = fast_span_estimate(K,Y,alpha,b,C)
%
% Compute an estimate of the number of errors made by leave-one-out 
% procedure for an SVM with L1 penalization of the slacks. 
% It only requires a matrix inversion and hence the complexity is not more
% than the SVM training itself.
%
% Input parameters :
%   K : kernel matrix
%   Y : labels (-1,1)
%   alpha,b : lagrange multipliers and threshold
%   C : soft margin paramater, aka upper bound on alpha.
%
% Output parameters :
%   loo = estimate of the fraction of leave-one-out errors.
%
% Source: http://olivier.chapelle.cc/ams/

span = zeros(size(K,1),1);

% Compute the outputs on the training points
output = Y.*(K*(alpha.*Y)+b);

% Find the indices of the support vectors of first and second category
eps = 1e-5;
sv1 = find(alpha > max(alpha)*eps & alpha < C*(1-eps));
sv2 = find(alpha > C*(1-eps));


% Degenerate case: if sv1 is empty, then we assume nothing changes 
% (loo = training error)
if isempty(sv1)
  loo = mean(output < 0);
  return;
end;

% Compute the invert of KSV
l = length(sv1);
KSV = [[K(sv1,sv1) ones(l,1)]; [ones(1,l) 0]];
% a small ridge is added to be sure that the matrix is invertible
invKSV=inv(KSV+diag(1e-12*[ones(1,l) 0]));

% Compute the span for all support vectors.
n = length(K);     % Number of training points
span = zeros(n,1); % Initialize the vector
tmp = diag(invKSV);
span(sv1) = 1./tmp(1:l); % Span of sv of first category
% If there exists sv of second category, compute their span
if ~isempty(sv2)  
  V = [K(sv1,sv2); ones(1,length(sv2))];
  span(sv2) = diag(K(sv2,sv2)) - diag(V'*invKSV*V);
end;

% Estimate the fraction of loo error
loo = mean(output - alpha.*span < 0);