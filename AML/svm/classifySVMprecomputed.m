function [y_class score] = classifySVMprecomputed(K,y,model)
%this kernel must be TS * TR'
%ogni riga contiene il K di un campione di TS con tutti quelli di TR.

[y_class, accuracy, score] = svmpredict(y,[(1:size(K,1))' K], model);

%se non hai prob in uscita
score = score.*model.Label(1); %cosi alla classe -1 da sempre score negativi

%score = score(:,model.Label==1);