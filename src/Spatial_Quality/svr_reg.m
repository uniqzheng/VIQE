function [lcc, srocc, rmse ] = svr_reg( features, scores, Iterations )
% features is a matrix of size M x N; where M is total number of features
% and N is total number of feature elements
% scores is M x 1 matrix 

temp=[];
for iter = 1:1:Iterations
    [trainInd,valInd,testInd] = dividerand(length(features),0.8,0,0.2);
    mdl = fitrsvm(features(trainInd,:),scores(trainInd),'KernelFunction','rbf','KernelScale','auto','Standardize',true,'CacheSize','maximal');
    y_cap = predict(mdl,features(testInd,:));
    mos_cap = scores(testInd);
    temp(iter,:) = calculatepearsoncorr(y_cap,mos_cap);
    median(temp);
end
result = median(temp);
lcc = result(1);srocc = result(2); rmse = result(3);
end

