function plotSVMprecomputed(x,y,model)

SV_idx=full(model.SVs);

plot(x(y==1,1),x(y==1,2),'r.','MarkerSize',12); hold on;
plot(x(y==-1,1),x(y==-1,2),'b.','MarkerSize',12);
grid on

text(x(SV_idx,1)+0.05,x(SV_idx,2)+0.05,num2str(abs(model.sv_coef),2));

minX=-4;%min(x(:,1));
maxX=+4;%max(x(:,1));
minY=-4;%min(x(:,2));
maxY=+4;%max(x(:,2));

%creates a grid of n x n points
n=200;
[bigX, bigY] = meshgrid(minX:(maxX-minX)/(n-1):maxX, minY:(maxY-minY)/(n-1):maxY);

ntest=size(bigX, 1) * size(bigX, 2);
instance_test=[reshape(bigX, ntest, 1), reshape(bigY, ntest, 1)];
label_test = zeros(size(instance_test, 1), 1);

%[tmp, acc, Z] = svmpredict(label_test, instance_test, model);
%
%
%che kernel?
K=lin(x,instance_test);


[tmp Z] = classifySVMprecomputed(K,label_test,model);
bigZ = reshape(Z, size(bigX, 1), size(bigX, 2));

[C h] = contour(bigX, bigY, bigZ, [-1 0 1], '--k'); %clabel(C,h);
contour(bigX, bigY, bigZ, [0 0], 'k');

axis([min(x(:,1)) max(x(:,1)) min(x(:,2)) max(x(:,2))])

