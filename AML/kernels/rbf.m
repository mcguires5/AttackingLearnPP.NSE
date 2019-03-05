%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the matrix of kernel values between x_ts and x_tr
%
% Input parameters :
%   x_ts : matrix containing the first set of points. Representing the
%          validation set, each row is an observation and 
%          each column is a feature of the data 
%   x_tr : matrix containing the second set of points. Representing the
%         training set, each row is an observation and 
%         each column is a feature of the data 
%   G : Gamma scalar value used with RBF kernel
% Output parameters :
%   K : matrix containing the kernel values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function K = rbf(x_ts,x_tr,G)

K =  -2*(x_ts*x_tr');
col_add = diag(x_ts*x_ts');
row_add = diag(x_tr*x_tr');
for k=1:size(x_ts,1)
   K(k,:)=K(k,:)+row_add';
end
for k=1:size(x_tr,1)
   K(:,k)=K(:,k)+col_add;
end
K = exp(-G*K);


