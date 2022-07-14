%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modification I made to the code to get started on regime switching models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars;
% addpath([cd '/subroutines'])  %on a MAC
% addpath([cd '/subroutines/DERIVESTsuite'])  %on a MAC
% addpath([cd '/subroutines_additional'])  %on a MAC
addpath([cd '\subroutines']) %on a PC
addpath([cd '\subroutines/DERIVESTsuite'])  %on a PC
addpath([cd '\subroutines_additional'])  %on a PC
addpath([cd '\functions'])

figures_dir = 'C:\Users\Jota\Dropbox\Aplicaciones\Overleaf\Econ722-Term-Paper';

run_extra_code = 0; 


%% load the monthly macro dataset 
%[~,TEXTCPI,~] = xlsread('dataMLprojectMay2021.xlsx','Monthly');
DATAMACRO = readmatrix('dataMLprojectMay2021.xlsx');  
TimeMACRO = datetime(datestr((DATAMACRO(:,1)+datenum('12/31/1899','mm/dd/yy'))));
DataMACRO=DATAMACRO(:,2:end);

% some data transformations
DataMACRO(:,15)=exp(DataMACRO(:,15)/100);           % unemployment (because all variables are then logged and multiplied by 100)
DataMACRO(:,16)=exp(DataMACRO(:,16)/100);           % GZ spread (because all variables are then logged and multiplied by 100)
DataMACRO(:,9)=DataMACRO(:,9)./DataMACRO(:,12);     % real pce, nominal PCE/PCE deflator
DataMACRO(:,14)=DataMACRO(:,14)./DataMACRO(:,6);    % real pce services, nominal PCE services/PCE services deflator

% the data are ordered as
% [ 1      2      3       4        5       6     7   8     9    10      11     12      13      14   15     16    ]
% [cpi  ppcedg  ppceg  ppcendg  corepce  ppces  ip  empl  pce  pcedg  pcendg  ppce  coreppce  pces unem  GZspread]

% variables in the baseline model
indmacro=[15 8 9 14 12 6 13]; 
series=["unemployment","employment","PCE","PCE: services","PCE (price)","PCE: services (price)","core PCE (price)"];
YLABELirf=["percentage points","100 x log points","100 x log points","100 x log points","100 x log points","100 x log points","100 x log points"];
YLABELfcst=["percentage points","index","index","index","index","index","index"];
 
 
%% choice of estimation sample, constant or varying volatility, and forecasting period

T0 = find(year(TimeMACRO)==1988 & month(TimeMACRO)==12);        % beginning of estimation sample
T1estim = find(year(TimeMACRO)==2021 & month(TimeMACRO)==5);    % end of estimation sample

T1av = find(year(TimeMACRO)==2021 & month(TimeMACRO)==5);       % date of last available data for forecasting
Tend = find(year(TimeMACRO)==2021 & month(TimeMACRO)==5);       % date of last available data in the dataset

Tfeb2020 = find(year(TimeMACRO)==2020 & month(TimeMACRO)==2);   % Position of the Feb 2020 observation (should not be modified)
Tcovid=[];                                                      % first time period of COVID (March 2020; should be set to [] if constant volatility)

Tjan2019=Tfeb2020-13;                                           % initial date for conditional forecast plots (no need to modify)
TendFcst=Tfeb2020+22+6;                                         % end date for projections (June 2022)
hmax=TendFcst-T1av;                                             % corresponding maximum forecasting horizon     


%% monthly VAR estimation
Ylev = DataMACRO(T0:T1estim,indmacro);
Ylog = 100*log(Ylev);
Time = TimeMACRO(T0:end);
[~,n] = size(Ylog);

rng(10);            % random generator seed
lags=13;            % # VAR lags
ndraws=2*2500;      % # MCMC draws
res = bvarGLP_covid(Ylog,lags,'mcmc',1,'MCMCconst',1,'MNpsi',0,'sur',0,'noc',...
    0,'Ndraws',ndraws,'hyperpriors',1,'Tcovid',Tcovid);

% This is irrelevant here because this is constant volatility
disp(res.postmax.eta)

% Display the variance and covariance matrix
disp(res.postmax.sigmahat)
writematrix(tril(round(res.postmax.sigmahat,4)),...
    [figures_dir '\change_2_variance_baseline.csv']); 

%% Simple regime switching model to get started

% Sort the data
% CPI_logs = log(DataMACRO(:,1)); 
% CPI_logs_prev = CPI_logs(1:(end-1)); 
% inflation = CPI_logs(2:end) - CPI_logs_prev; 
% clear CPI_logs CPI_logs_prev 

opts = delimitedTextImportOptions("NumVariables", 2);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["DATE", "USAGDPDEFQISMEI"];
opts.VariableTypes = ["datetime", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "DATE", "InputFormat", "yyyy-MM-dd");

% Import the data
inflationdataUSAGDPDEFQISMEI = readtable("inflation_data_USAGDPDEFQISMEI.csv", opts);


% Clear temporary variables
clear opts
 
% Import the data
dates = table2array(inflationdataUSAGDPDEFQISMEI(:,1)); 
tbl = table2array(inflationdataUSAGDPDEFQISMEI(:,2));
CPI_logs = log(tbl); 
CPI_logs_prev = CPI_logs(1:(end-1)); 
inflation = CPI_logs(2:end) - CPI_logs_prev; 
clear CPI_logs CPI_logs_prev inflationdataUSAGDPDEFQISMEI tbl 
 
 
% Try filter function - Filtering + Smoothing 
[~, ~, regime_0_like_cond, regime_1_like_cond, likeli_vector, ...
    regime_smooth, q0_smooth, q1_smooth, st_given_next] = ...
    filter_example...
    (0.001, 0.001, 0.9, 0.75, 0.005, 0.95, 0.8, inflation, 1); 
  
% Try regression function 
[bbeta,~] = regression_example(inflation, regime_smooth);

 

% Now do some simple MCMC

% Places to store things
df = length(inflation) - 3; 
N = 5000; 
aalpha_post = zeros(N,2);
regime_post = zeros(N,length(inflation)); 

% Hyperparameters and initialization 

for ii = 1:N
    
    % Get some random draws
    %variance = iwishrnd([1,0;0,1], df); 

    % Get some random draws for alpha
    aalpha = mvnrnd([0.001; 0.001], 10^(-5)*[1, 0; 0,1]); 

    % Filter and Smooth
    [~, ~, ~, ~, ~, ...
    regime_smooth, ~, ~, ~] = filter_example(aalpha(1), aalpha(2), 0.9, 0.75, ...
    0.005, 0.95, 0.8, inflation, 1); 

     

    % Regression 
    [aalpha_post(ii, :), ~] = regression_example(inflation, regime_smooth); 
    regime_post(ii, :) = transpose(regime_smooth); 
end

% Keep only 2500 last draws and prepare to plot
T = length(inflation); 
q_regime_prob = transpose(sum(regime_post(2501:end,:))/2500); 
q_regime = zeros(T,1);
mean_to_plot = mean(aalpha_post(2501:end,1),'omitnan') * ones(length(inflation),1);
temp_to_add = mean(aalpha_post(2501:end,2),'omitnan');

for tt = 1:T

    if q_regime_prob(tt) > 0.5

        q_regime(tt) = 1;
        mean_to_plot(tt) = mean_to_plot(tt) + temp_to_add;


    end

end

yyyy = year(dates);
yyyy = yyyy(2:end); 

figure(1)
plot(yyyy, inflation, 'LineWidth', 2, 'Color', 'r')
hold on
plot(yyyy, mean_to_plot, 'LineWidth', 2, 'Color', 'b')
legend('Inflation', 'Mean', 'Location', 'best')
title('Different Regimes for the Inflation Rate')
ylabel('Inflation')
xlabel('Date')
saveas(gcf, [figures_dir '\change_2_inflation_regime.png'])
close(figure(1))

%% Regime switching model for the variance with simulated data
clearvars -except run_extra_code figures_dir 
clc 

% Simulate some data 
T = 200;
x_axis = transpose(1:T); 
mmu = 0;
ssigma = 1; 
data = normrnd(mmu, ssigma, [T,1]); 
data_hv = normrnd(mmu, 4*ssigma, [floor(3*T/4)-floor(T/2),1]);
data((floor(T/2)+1):floor(3*T/4)) = data_hv; 
clear data_hv 
  
% Try the function 
[log_like, regime_cond, ~, ~, ~, ...
    regime_smooth, ~, ~, ~] = ...
    filter_example...
    (0.001, 0, 0.9, 0.75, 1, 0.95, 0.8, data, 2); 

% Transform the data 
transformed_data = (data - mean(data)).^2; 

% Try regression 
[bbeta_hat, ssigma_hat] = regression_example(transformed_data, regime_smooth); 

disp(mean(regime_cond))
disp(mean(regime_smooth))
disp(log_like)
disp(bbeta_hat)
disp(ssigma_hat)
clear regime_cond regime_smooth log_like bbeta_hat ssigma_hat mmu ssigma 
    

% Now do mcmc and see what happens 
N = 5000; 
regime_post = zeros(N,T); 

for ii = 1:N 

    % Get some random draws from the deviation 
    ssigma = 1/gamrnd(1,1); 
    
    % Filter and Smooth
    [~, ~, ~, ~, ~, ...
    regime_post(ii, :), ~, ~, ~] = filter_example(0.01, 0, 0.9, 0.75, ssigma, 0.95, ...
    0.8, data, 2); 
   
end 

% Keep last 2500 draws 
q_regime_prob = transpose(sum(regime_post(2501:end,:))/2500); 
q_regime = zeros(T,1);
 
for tt = 1:T

    if q_regime_prob(tt) > 0.5

        q_regime(tt) = 1;

    end

end

% Plot 
figure(2)
plot(x_axis, max(data)*q_regime, 'Color', [1 1 1]*0.9)
hold on 
plot(x_axis, min(data)*q_regime, 'Color', [1 1 1]*0.9)
x2 = [x_axis, fliplr(x_axis)];
inBetween = [max(data)*q_regime, fliplr(min(data)*q_regime)];
fill(x2, inBetween, 0.9*[1 1 1]);
plot(x_axis, data, 'LineWidth', 1, 'Color', 'r')
%legend('Data', 'High Volatility', 'Location', 'best')
title('Different Regimes for Simulated Data')
ylabel('Data')
xlabel('Time') 
saveas(gcf, [figures_dir '\change_2_volatility_regime.png'])
close(figure(2))

 

disp(sqrt(mean(transformed_data)))
disp(sqrt(mean(transformed_data(q_regime==1))))
disp(sqrt(mean(transformed_data(q_regime==0))))
disp(sqrt(mean(transformed_data(q_regime==1)))/sqrt(mean(transformed_data(q_regime==0))))






%% generalized IRFs to an "unemployment" shock

if run_extra_code == 1

H=60;
M = size(res.mcmc.beta,3);
Dirf1 = zeros(H+1,size(Ylog,2),M);
for jg = 1:M
    Dirf1(:,:,jg) =  bvarIrfs(res.mcmc.beta(:,:,jg),res.mcmc.sigma(:,:,jg),1,H+1);
end
sIRF1 = sort(Dirf1,3);


%% conditional forecasts
YYfcst=[100*log(DataMACRO(Tjan2019:T1av,indmacro));NaN(hmax,n)];  
YYfcst(end-hmax+1:end,1)=4+(5.8-4)*.85.^(0:hmax-1)';                % conditioning scenario from Blue Chip

TTfcst=length(YYfcst);
DRAWSY=NaN(n,TTfcst,M);      % matrix to store draws of variables
% Forecasts
for i=1:M
    betadraw=squeeze(res.mcmc.beta(:,:,i));
    G=chol(squeeze(res.mcmc.sigma(:,:,i)))';
    if isempty(Tcovid)
        etapar=[1 1 1 1];
        tstar=1000000;
    else
        etapar=res.mcmc.eta(i,:); 
        tstar=TTfcst-hmax+Tcovid-T;
    end
    [varc,varZ,varG,varC,varT,varH]=FormCompanionMatrices(betadraw,G,etapar,tstar,n,lags,TTfcst);
    s00=flip(YYfcst(1:lags,:))'; s00=s00(:);
        
    P00=zeros(n*lags,n*lags);
    [DrawStates,shocks]=DisturbanceSmootherVAR(YYfcst,varc,varZ,varG,varC,varT,varH,s00,P00,TTfcst,n,n*lags,n,'simulation');
    DRAWSY(:,:,i)=DrawStates(1:n,:);
end
IRFA=DRAWSY(1:n,:,:);
IRFAsorted=sort(IRFA,3);


%% plot of conditional forecasts
qqq=[.025 .16 .5 .84 .975];         % percentiles of the posterior distribution for plots
ColorCovid=[.8941, .1020, .1098];   % colors for plots
ColorBase=[44,127,184]./255;
ColorGrey=[.5 .5 .5];


%% saving the results
cd results
save CV_May2021
cd ..

end
    
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% additional function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [varc,varZ,varG,varC,varT,varH]=FormCompanionMatrices(betadraw,G,etapar,tstar,n,lags,TTfcst)
% forms the matrices of the VAR companion form

% matrices of observation equation
varc=zeros(n,TTfcst);
varZ=zeros(n,n*lags); varZ(:,1:n)=eye(n); varZ=repmat(varZ,1,1,TTfcst);
varG=repmat(zeros(n),1,1,TTfcst);

% matrices of state equation
B=betadraw;
varC=zeros(n*lags,1); varC(1:n)=B(1,:)';
varT=[B(2:end,:)';[eye(n*(lags-1)) zeros(n*(lags-1),n)]];
varH=zeros(n*lags,n,TTfcst); 
for t=1:TTfcst
    if t<tstar
        varH(1:n,1:end,t)=G;
    elseif t==tstar
        varH(1:n,1:end,t)=G*etapar(1);
    elseif t==tstar+1
        varH(1:n,1:end,t)=G*etapar(2);
    elseif t==tstar+2
        varH(1:n,1:end,t)=G*etapar(3);
    elseif t>tstar+2
        varH(1:n,1:end,t)=G*(1+(etapar(3)-1)*etapar(4)^(t-tstar-2));
    end
end
end

