%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Regime Switching Code 
%
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

% Load data ---------------------------------------------------------------
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

% choice of estimation sample, constant or varying volatility, and forecasting period

T0 = find(year(TimeMACRO)==1988 & month(TimeMACRO)==12);        % beginning of estimation sample
T1estim = find(year(TimeMACRO)==2021 & month(TimeMACRO)==5);    % end of estimation sample

T1av = find(year(TimeMACRO)==2021 & month(TimeMACRO)==5);       % date of last available data for forecasting
Tend = find(year(TimeMACRO)==2021 & month(TimeMACRO)==5);       % date of last available data in the dataset

Tfeb2020 = find(year(TimeMACRO)==2020 & month(TimeMACRO)==2);   % Position of the Feb 2020 observation (should not be modified)
Tcovid=[];                                                      % first time period of COVID (March 2020; should be set to [] if constant volatility)

Tjan2019=Tfeb2020-13;                                           % initial date for conditional forecast plots (no need to modify)
TendFcst=Tfeb2020+22+6;                                         % end date for projections (June 2022)
hmax=TendFcst-T1av;                                             % corresponding maximum forecasting horizon     


% monthly VAR estimation
Ylev = DataMACRO(T0:T1estim,indmacro);
Ylog = 100*log(Ylev);
Time = TimeMACRO(T0:end);
[~,n] = size(Ylog);

rng(10);            % random generator seed
lags=13;            % # VAR lags
ndraws=2*2500;      % # MCMC draws

clear DataMACRO DATAMACRO 

% res = bvarGLP_covid(Ylog,lags,'mcmc',1,'MCMCconst',1,'MNpsi',0,'sur',0,'noc',...
    % 0,'Ndraws',ndraws,'hyperpriors',1,'Tcovid',Tcovid);

% First try to evaluate the likelihood
    % I want to run bvarGLP_covid and get the likelihood from that script
    % since I know it works --> No, I don't want to do this. I don't want
    % to maximize likelihood everytime. Remember I want to feed the
    % algorithm with some true parameter values 
    % If Tcovid is empty, then this is doing constant volatility. This
    % works and I do not want to change that. 
    % Now if Tcovid is not empty, maybe that is the place I want to change
    % the code and modify things to estimate an extra parameter 
    % Let's see if I can understand logMLVAR_formin_covid.m 
% res = bvarGLP_covid(Ylog,lags,'mcmc',0,'MCMCconst',1,'MNpsi',0,'sur',0,'noc',...
     % 0,'Ndraws',ndraws,'hyperpriors',1,'Tcovid',Tcovid);
% disp(res.postmax.logPost)

% Set priors to see what is going on
numarg = 1; 
setpriors_covid; 

% Construct matrix of regressors 
[TT,~]=size(Ylog);
k=n*lags+1;  
x=zeros(TT,k);
x(:,1)=1;
for i=1:lags
    x(:,1+(i-1)*n+1:1+i*n)=lag(Ylog,i);
end

y0=mean(Ylog(1:lags,:),1);
x=x(lags+1:end,:);
y=Ylog(lags+1:end,:);
[T,~]=size(y);

SS=zeros(n,1);
for i=1:n
    Tend=T; if ~isempty(Tcovid); Tend=Tcovid-1; end
    ar1=ols1(y(2:Tend,i),[ones(Tend-1,1),y(1:Tend-1,i)]);
    SS(i)=ar1.sig2hatols;
end

b=zeros(k,n);
diagb=ones(n,1);
diagb(pos)=0;   % Set to zero the prior mean on the first own lag for variables selected in the vector pos
b(2:n+1,:)=diag(diagb);

% The following function evaluates the posterior of the hyperparameters 
    % 1) is lambda 
    % 2) is theta
    % 3) is mu 
    % In between covid paramters (if there are any)
    % Last is alpha 

% I'm gonna let Tcovid = 0.5 to modify that code. This means we are adding
% one extra parameter to capture higher volatility 

[logML, ~, ~] = logMLVAR_formin_covid([0.5;0.5;0.5;2;0.5], y, x, lags, ...
    T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, 0.5); 
disp(logML)
[logML, bbetahat, ssigmahat] = logMLVAR_formin_covid([0.5;0.5;0.5;0.5], y, x, lags, ...
    T, n, b, MIN, MAX, SS, Vc, pos, mn, sur, noc, y0, hyperpriors, priorcoef, []); 
disp(logML)


