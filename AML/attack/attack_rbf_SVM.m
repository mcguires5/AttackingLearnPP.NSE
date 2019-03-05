%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function simulating an attack starting at a given points for SVMs 
% trained with a RBF kernel.
% The attack is described at Algorithm 1 in the article Poisoning Attack
% Against SVMs.
% First a SVM is trained on the training data set plus the attack point
% Then the attack point is shifted in the feature space and its influence
% computed.
%
% Input parameters :
%   step : scalar representing the step size of the attack (gradient
%          method)
%   xc : row vector representing the initial position of the attack point
%   yc : integer which is the label associated to the attack point
%   lb : scalar representing the lower bound of the box constraint of final
%        figures
%   ub : scalar representing the upper bound of the box constraint of final
%        figures
%   x_tr : matrice of the training data set. Rows are observations and each
%          column is a feature defining data.
%   y_tr : column vector representing labels associated to training data
%   x_vd : matrice of the evaluation data set. Rows are observations and each
%          column is a feature defining data.
%   y_vd : column vector representing labels associated to evaluation data
%   C : scalar representing the cost incurred by the classifier when it
%       makes errors
%   Gamma : Gamma scalar value used with RBF kernel
% Output parameters :
%   xc : the last position computed by the attack
%   err : the number of errors performed by the classifier
%   xc_seq : matrice representing values taken by the attack point over the
%            attack
%   alphac_seq : column vector representing values that \alpha_c has
%                taken over the attack
%   xi_obj : column vector containing the value of the hinge loss function
%            at each iteration of the attack
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xc err xc_seq alphac_seq xi_obj]= attack_rbf_SVM(step, xc, yc, lb, ub, x_tr, y_tr, x_vd, y_vd, C, Gamma)

one_class = (numel(unique(y_tr)) == 1);

% train a SVM on the new training set and retrieve Support Vector indices
[model alpha] = trainSVM(x_tr,y_tr,C,Gamma);
margin_SV_idx = find(alpha > 1E-6 & alpha < C-1E-6);
if(one_class) %recall that we set C=nu in one class
    margin_SV_idx=find(alpha > 1E-6 & alpha < 1/(C*size(x_tr,1)) -1E-6);
end

% evaluation the performance of the trained SVM
[yclass score]=classifySVM(x_vd,y_vd,model);
err(1) = sum(yclass~=y_vd)/size(x_vd,1);
disp(['Validation error (%): ' num2str(100*err(1))]);

% compute the influence of the attack point on the errors
g = y_vd.*score -1;
if(one_class==1)
    g=g+1;
end
xi_obj(1) = sum(max(0,-g));

%update output variables
alphac_seq(1) = alpha(end);
xc_seq(1,:)=xc;

%kernel matrix for TR and VD
Q=rbf(x_tr,x_tr,Gamma).*(y_tr*y_tr');
Q_tst = rbf(x_vd,x_tr,Gamma).*(y_vd*y_tr');

for i=1:500
    
    %update last row and last column
    Qic = rbf(x_tr,xc,Gamma).*(y_tr*yc);
    Q(:,end)=Qic';
    Q(end,:)=Qic;
    
    %compute attack direction and update the attack point's position
    dxc = attackSVMshiftRBF(x_tr,y_tr,x_tr(end,:),yc,margin_SV_idx,g,Q,Q_tst,x_vd,y_vd,alpha(end),Gamma); 
    xc = xc+step*dxc;
    
    %projection on box
    xc(xc>ub)=ub;
    xc(xc<lb)=lb;

    x_tr(end,:) = xc;

    xc_seq(i+1,:)=xc;
    
    % train a SVM on the updated training set and retrieve Support Vector indices
    [model alpha] = trainSVM(x_tr,y_tr,C,Gamma);
    margin_SV_idx = find(alpha > 1E-6 & alpha < C-1E-6);
    if(one_class)
        margin_SV_idx=find(alpha > 1E-6 & alpha < 1/(C*size(x_tr,1)) -1E-6);
    end

     % evaluation the performance of the trained SVM
    [yclass score]=classifySVM(x_vd,y_vd,model);
    err(i+1) = sum(yclass~=y_vd)/size(x_vd,1);

    % compute the influence of the attack point on the errors
    g = y_vd.*score -1;
    if(one_class==1)
        g=g+1;
    end

    %update output variables
    xi_obj(i+1) = sum(max(0,-g));
    
    alphac_seq(i+1) = alpha(end);
    
    disp(['Validation error (%): ' num2str(100*err(i+1))]);

    % after 50 iterations, if the number of errors performed is less than a
    % threshold (10^-6), the attack is stopped
     if(i>50 && err(i+1)-err(i-50) <= 1E-6)
         return;
     end
    
end


