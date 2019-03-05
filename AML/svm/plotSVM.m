%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function allowing to plot a SVM on a pre-created figure
% the plot will contain the training points, the separating functions and
% the margins
%
% Input parameters :
%   x : a matrix containing a set of points. A row represents an
%       observation and each column is a feature
%   y : column vector representing labels associated to data
%   model : a precomputed solution of a SVM
%   grid_coords : grid coordinate to plot a background function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotSVM(x,y,model,grid_coords)

if(nargin < 4)
    grid_coords = [min(x(:,1)) max(x(:,1)) min(x(:,2)) max(x(:,2))];
end

%retrieve Support Vectors from the model
[tmp SV_idx]=ismember(model.SVs, x,'rows');

%plot points from the first class (positive class) in red
%and points from the second class (negative class) in blue
plot(x(y==1,1),x(y==1,2),'r.','MarkerSize',28); hold on;
plot(x(y==-1,1),x(y==-1,2),'b.','MarkerSize',28);
grid on

%put labels with alpha values for each SV
%text(x(SV_idx,1)+0.05,x(SV_idx,2)+0.05,num2str(abs(model.sv_coef),2));
plot(x(SV_idx,1),x(SV_idx,2),'kO','MarkerSize',12);


minX= grid_coords(1);
maxX= grid_coords(2);
minY= grid_coords(3);
maxY= grid_coords(4);

%creates a grid of n x n points
n=200;
[bigX, bigY] = meshgrid(minX:(maxX-minX)/(n-1):maxX, minY:(maxY-minY)/(n-1):maxY);

ntest=size(bigX, 1) * size(bigX, 2);
instance_test=[reshape(bigX, ntest, 1), reshape(bigY, ntest, 1)];
label_test = zeros(size(instance_test, 1), 1);

%background function
[tmp, acc, Z] = svmpredict(label_test, instance_test, model);
bigZ = reshape(Z, size(bigX, 1), size(bigX, 2));

[C h] = contour(bigX, bigY, bigZ, [-1 1], 'k:'); %clabel(C,h);
contour(bigX, bigY, bigZ, [0 0], 'k');

axis(grid_coords)

