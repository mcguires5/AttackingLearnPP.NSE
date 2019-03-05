function [model alpha b SV_idx C] = trainSVMprecomputed(K,y,weights,C)
%Trains a SVM using a precomputed kernel matrix

if(nargin==2)
    weights='';
end

addpath libsvm

K1 = [(1:size(K,1))', K]; %we are required to add a column with sample idx

if(nargin == 3)
    %se C non e' definito, fa xval
    bestcv = 0;
    for log10c = -3:3,
        cmd = ['-v 5 -h 0 -t 4 ' weights ' -c ', num2str(10^log10c)];
        cv = svmtrain(y, K1, cmd);
        if (cv >= bestcv),
          bestcv = cv; bestc = 10^log10c;
        end
        fprintf('%g %g (best c=%g, rate=%g)\n', log10c, cv, bestc, bestcv);
    end

    C=bestc;
end

model = svmtrain(y,K1,['-h 0 -t 4 ' weights ' -c ' num2str(C)]);

SV_idx = full(model.SVs);

alpha = zeros(size(K,1),1);
alpha(SV_idx)=abs(model.sv_coef);

b = -model.rho;