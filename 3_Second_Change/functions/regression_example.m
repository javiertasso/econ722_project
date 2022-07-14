function [bbeta, ssigma] = regression_example(data_ts, regime_smooth)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

T = length(data_ts); 
y = data_ts; 
X = [ones(T,1),regime_smooth]; 
bbeta = (transpose(X) * X) \ (transpose(X) * y);
yhat = X * bbeta; 
sst = transpose(yhat) * yhat; 
ssigma = sst / (T - length(bbeta)); 

end