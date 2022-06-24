%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VAR estimation with CONSTANT volatility.
% Estimation sample ends in February 2020.
% Forecasts are produced starting in June 2021. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
addpath([cd '/subroutines'])  %on a MAC
addpath([cd '/subroutines/DERIVESTsuite'])  %on a MAC
addpath([cd '/subroutines_additional'])  %on a MAC
% addpath([cd '\subroutines']) %on a PC
% addpath([cd '\subroutines/DERIVESTsuite'])  %on a PC
% addpath([cd '\subroutines_additional'])  %on a PC


%% load the monthly macro dataset 
[DATAMACRO,TEXTCPI] = xlsread('dataMLprojectMay2021.xlsx','Monthly');
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
T1estim = find(year(TimeMACRO)==2020 & month(TimeMACRO)==2);    % end of estimation sample

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
[T,n] = size(Ylog);

rng(10);            % random generator seed
lags=13;            % # VAR lags
ndraws=2*2500;      % # MCMC draws
res = bvarGLP_covid(Ylog,lags,'mcmc',1,'MCMCconst',1,'MNpsi',0,'sur',0,'noc',0,'Ndraws',ndraws,'hyperpriors',1,'Tcovid',Tcovid);


%% generalized IRFs to an "unemployment" shock
H=60;
M = size(res.mcmc.beta,3);
Dirf1 = zeros(H+1,size(Ylog,2),M);
for jg = 1:M
    Dirf1(:,:,jg) =  bvarIrfs(res.mcmc.beta(:,:,jg),res.mcmc.sigma(:,:,jg),1,H+1);
end
sIRF1 = sort(Dirf1,3);


%% conditional forecasts
YYfcst=[100*log(DataMACRO(Tjan2019:T1av,indmacro));NaN(hmax,n)];  
YYfcst(end-hmax+1:end,1)=4+(5.8-4)*.85.^[0:hmax-1]';                % conditioning scenario from Blue Chip

TTfcst=length(YYfcst);
DRAWSY=NaN(n,TTfcst,M);      % matrix to store draws of variables
% Forecasts
for i=1:M
    betadraw=squeeze(res.mcmc.beta(:,:,i));
    G=chol(squeeze(res.mcmc.sigma(:,:,i)))';
    if isempty(Tcovid);
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
save CVFeb2020_May2021
cd ..

    
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% additional function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [varc,varZ,varG,varC,varT,varH]=FormCompanionMatrices(betadraw,G,etapar,tstar,n,lags,TTfcst);
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