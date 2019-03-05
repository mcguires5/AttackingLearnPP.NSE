%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the direction of the attack to make the classifier performs as
% much errors as possible. This function only applies in the case of the
% use of a linear kernel.
%
% Input parameters :
%   x : a matrix containing the training support vectors. A row represents
%       an observation and each column would be a feature of the data.
%   y : column vector representing labels associated to training data
%   xc : row vector representing the initial position of the attack point
%   yc : integer which is the label associated to the attack point
%   margin_SV_idx : a column vector containing the indices of points which
%                   are on the margins of the solution given by the computed SVM.
%                   (Support vectors are points for which
%                   their alpha value is > 0 and < C)
%   g : a column vector representing the score of the class assigned to an
%       example
%   Q : a matrice representing the gradient of the rbf kernel. This matrix
%       has a size of size(x)*size(x) 
%   Q_tst : a matrice representing the gradient of the rbf kernel. This
%           matrix has a size of size(x_tst)*size(x)
%   x_tst : matrice of the evaluation data set. Rows are observations and each
%           column is a feature defining data.
%   y_tst : column vector representing labels associated to evaluation data
%   alphac : a column vector containing the alpha value of the attack point
%   Gamma : Gamma scalar value used with RBF kernel
% Output parameters :
%   dxc : A column representing the computed direction of the attack.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dxc  = attackSVMshiftRBF(x,y,xc,yc,margin_SV_idx,g,Q,Q_tst,x_tst,y_tst,alphac,Gamma)

% if there is no support vectors on the margins of the previous solution of
% SVM, do not move the attack point
if(isempty(margin_SV_idx))
    disp('No margin SVs!')
    dxc=zeros(1,size(xc,2));
    return;
end

%compute eq 7 from the article Poisoning Attacks Against SVMs
Q= Q+1E-6*eye(size(Q,1));
R = [0 y(margin_SV_idx)'; y(margin_SV_idx) Q(margin_SV_idx,margin_SV_idx)]^-1;

%compute eq 7 from the article Poisoning Attacks Against SVMs
%RBF kernel derived wrt xc
dQic = repmat(y_tst,1,size(x,2)).*rbf_dQ(xc,x_tst,Gamma);
dQsc = repmat(y(margin_SV_idx),1,size(x,2)).*rbf_dQ(xc,x(margin_SV_idx,:),Gamma); 

delta = -R*[zeros(1,size(dQsc,2)); dQsc];

%db and da needs to be multiply by alpha_c to match their definition in the
%article
db = delta(1,:);
da = delta(2:end,:);

Qis = Q_tst(:,margin_SV_idx);

%compute eq 3 from the article Poisoning Attacks Against SVMs
% bug fixed in v1.1 thanks to Nathalie Baracaldo and Jaehoon Safavi
delta_gi = Qis*da + y_tst*db + dQic*alphac;
delta_gi(g>=0,:)=0; %slack variables...

%dL/du can be computed from eq 10 or from the first definition of L (eq 1)
%since delta_gi is the derivate of g from u (eq 3)
%g is the sum of the negative dgi/du
%the direction is a norm-1 vector computed from the 2-norm
gradient = -sum(delta_gi);
dxc =  gradient / norm(gradient);

%if normalization did not go well, do not move the attack point
if(isnan(dxc))
    dxc=zeros(1,size(xc,2));
    return;
end
