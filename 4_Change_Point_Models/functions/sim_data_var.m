function data = sim_data_var(T, mmu, ssigma1, ssigma2)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

data = normrnd(mmu, ssigma1, [T,1]); 
data((floor(T/2)+1):T) = normrnd(mmu, ssigma2, [T-floor(T/2),1]);


end