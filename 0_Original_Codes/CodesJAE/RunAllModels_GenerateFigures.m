%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this code estimates all the different models used in the paper and 
% generates all the figures of the paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clear all

%% estimate all the models

% VAR with COVID volatility estimated on data up to May 2021
% forecasts produced starting in June 2021
Baseline_May2021

% VAR with CONSTANT volatility estimated on data up to Feb 2020
% forecasts produced starting in June 2021
CVFeb2020_May2021

% VAR with CONSTANT volatility estimated on data up to May 2021
% forecasts produced starting in June 2021
CV_May2021

% VAR with COVID volatility estimated on data up to June 2020
% forecasts produced starting in July 2020
Baseline_June2020

% VAR with CONSTANT volatility estimated on data up to Feb 2020
% forecasts produced starting in July 2020
CVFeb2020_June2020


%% produce the figures
GenerateFigures