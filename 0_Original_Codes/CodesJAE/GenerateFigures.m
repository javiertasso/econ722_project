%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code generates all the figures in the paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clear all
cd results


%% figure 1
f1=figure(1);
f1.Position = [0, 0, 700, 600];

load CVFeb2020_May2021
ColorPlot=ColorBase;
subplot(3,2,1:2); histogram(res.mcmc.lambda,20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
title('\lambda  (MN prior)')
if ~isempty(Tcovid)
    subplot(3,2,3); histogram(res.mcmc.eta(:,1),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\bar{s}_0$','Interpreter','Latex')
    subplot(3,2,4); histogram(res.mcmc.eta(:,2),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\bar{s}_1$','Interpreter','Latex')
    subplot(3,2,5); histogram(res.mcmc.eta(:,3),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\bar{s}_2$','Interpreter','Latex')
    subplot(3,2,6); histogram(res.mcmc.eta(:,4),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\rho$','Interpreter','Latex')
end

clear all
load CV_May2021
ColorPlot=ColorGrey;
subplot(3,2,1:2); histogram(res.mcmc.lambda,20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
title('\lambda  (MN prior)')
if ~isempty(Tcovid)
    subplot(3,2,3); histogram(res.mcmc.eta(:,1),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\bar{s}_0$','Interpreter','Latex')
    subplot(3,2,4); histogram(res.mcmc.eta(:,2),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\bar{s}_1$','Interpreter','Latex')
    subplot(3,2,5); histogram(res.mcmc.eta(:,3),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\bar{s}_2$','Interpreter','Latex')
    subplot(3,2,6); histogram(res.mcmc.eta(:,4),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\rho$','Interpreter','Latex')
end

clear all
load Baseline_May2021
ColorPlot=ColorCovid;
subplot(3,2,1:2); histogram(res.mcmc.lambda,20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
title('\lambda  (MN prior)')
if ~isempty(Tcovid)
    subplot(3,2,3); histogram(res.mcmc.eta(:,1),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\bar{s}_0$','Interpreter','Latex')
    subplot(3,2,4); histogram(res.mcmc.eta(:,2),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\bar{s}_1$','Interpreter','Latex')
    subplot(3,2,5); histogram(res.mcmc.eta(:,3),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\bar{s}_2$','Interpreter','Latex')
    subplot(3,2,6); histogram(res.mcmc.eta(:,4),20,'FaceAlpha',.6,'FaceColor',ColorPlot); hold on
    title('$\rho$','Interpreter','Latex')
end

subplot(3,2,1:2);
legend('constant volatility - sample ends in 2020:2','constant volatility - sample ends in 2021:5',...
      'Covid volatility - sample ends in 2021:5')
  
  
%% figure 2
f2=figure(2);
f2.Position = [0, 0, 700, 600];

clear all
load Baseline_May2021
ColorPlot=ColorCovid;
qqq=[.025 .16 .5 .84 .975];     % percentiles of the posterior distribution
count=0;
for jn = 1:n
    count=count+1;
    subplot(ceil(n/2),2,count)
    quantilePlot([0:H]', squeeze(sIRF1(:,jn,round(qqq*M)))/sIRF1(1,1,round(.5*M)),ColorPlot); hold on; grid on;
    xlabel('horizon');
    ylabel(YLABELirf(jn))
    title(series(jn));
end

clear all
load CVFeb2020_May2021
ColorPlot=ColorBase;
count=0;
for jn = 1:n
    count=count+1;
    subplot(ceil(n/2),2,count)
    plot([0:H]', squeeze(sIRF1(:,jn,round(.5*M)))/sIRF1(1,1,round(.5*M)),'color',ColorPlot,'LineWidth',2,'LineStyle','--'); hold on; grid on;
end

clear all
load CV_May2021
ColorPlot=ColorGrey;
count=0;
for jn = 1:n
    count=count+1;
    subplot(ceil(n/2),2,count)
    % plot([0:H]', squeeze(sIRF1(:,jn,round(.5*M))),'color',ColorPlot); hold on; grid on;
    plot([0:H]', squeeze(sIRF1(:,jn,round(.5*M)))/sIRF1(1,1,round(.5*M)),'color',ColorPlot,'LineWidth',2,'LineStyle','-.'); hold on; grid on;
        line([0 H],[0 0],'color','k')

end

% h = get(gca,'Children');
% set(gca,'Children',[h(1) h(2) h(3) h(5) h(6)])

legend('Covid volatility: posterior medians','Covid volatility: 68-percent credible regions','Covid volatility: 95-percent credible regions',...
    'constant volatility - sample ends in 2020:2: posterior medians','constant volatility - sample ends in 2021:5: posterior medians')


%% figure 3
clear all
f31=figure(31);
f31.Position = [0, 0, 350, 800];

load Baseline_June2020
ColorPlot=ColorCovid;
count=0;
for ii=1:n
    count=count+1;
    if ii~=1;
        aux=exp(squeeze(IRFAsorted(ii,:,round(qqq*M)))/100); normalization=aux(13,3); aux=100*aux/normalization; % normalization must be changed
        realization=100*exp(log(DataMACRO(T1av+1:Tend,indmacro(ii))))/normalization;
        
    elseif ii==1;
        aux=squeeze(IRFAsorted(ii,:,round(qqq*M)));
        realization=100*log(DataMACRO(T1av+1:Tend,indmacro(ii)));
    end
    subplot(7,1,count); quantilePlot([2019+.5/12:1/12:2022+5.5/12]', aux,ColorPlot); grid on; hold on  
    
    if ii~=1
        subplot(7,1,count); plot([2020+6.5/12:1/12:2021+4.5/12],realization,'k+','LineWidth',1.5)
    end
    
    xlim([2019+.5/12 2022+5.5/12])
    xticks([2019+.5/12 2020+.5/12 2021+.5/12 2022+.5/12])
    xticklabels({'Jan 2019','Jan 2020','Jan 2021','Jan 2022'})
    
    ylabel(YLABELfcst(ii));
    title(series(ii));
end
sgtitle('Covid volatility - est. sample ends in 2020:6','FontSize',12,'FontWeight','bold','color',ColorPlot)


clear all
f32=figure(32);
f32.Position = [0, 0, 350, 800];

load CVFeb2020_June2020
ColorPlot=ColorBase;
count=0;
for ii=1:n
    count=count+1;
    if ii~=1;
        aux=exp(squeeze(IRFAsorted(ii,:,round(qqq*M)))/100); normalization=aux(13,3); aux=100*aux/normalization; % normalization must be changed
        realization=100*exp(log(DataMACRO(T1av+1:Tend,indmacro(ii))))/normalization;
        
    elseif ii==1;
        aux=squeeze(IRFAsorted(ii,:,round(qqq*M)));
        realization=100*log(DataMACRO(T1av+1:Tend,indmacro(ii)));
    end
    subplot(7,1,count); quantilePlot([2019+.5/12:1/12:2022+5.5/12]', aux,ColorPlot); grid on; hold on  
    
    if ii~=1
        subplot(7,1,count); plot([2020+6.5/12:1/12:2021+4.5/12],realization,'k+','LineWidth',1.5)
    end
    
    xlim([2019+.5/12 2022+5.5/12])
    xticks([2019+.5/12 2020+.5/12 2021+.5/12 2022+.5/12])
    xticklabels({'Jan 2019','Jan 2020','Jan 2021','Jan 2022'})
    
    ylabel(YLABELfcst(ii));
    title(series(ii));
end
sgtitle('Constant volatility - est. sample ends in 2020:2','FontSize',12,'FontWeight','bold','color',ColorPlot)


for jj=1:n
    figure(31); subplot(7,1,jj); yl=get(gca,'ylim'); YMIN=yl(1); YMAX=yl(2);
    figure(32); subplot(7,1,jj); yl=get(gca,'ylim'); YMIN=[YMIN  yl(1)]; YMAX=[YMAX yl(2)];
    YLIM=[min(YMIN) max(YMAX)];
    figure(31); subplot(7,1,jj); set(gca,'ylim',YLIM); 
    figure(32); subplot(7,1,jj); set(gca,'ylim',YLIM); 
    clear YMIN YMAX YLIM
end


%% figure 4
clear all
f41=figure(41);
f41.Position = [0, 0, 350, 800];

load Baseline_May2021
ColorPlot=ColorCovid;
count=0;
for ii=1:n
    count=count+1;
    if ii~=1;
        aux=exp(squeeze(IRFAsorted(ii,:,round(qqq*M)))/100); normalization=aux(13,3); aux=100*aux/normalization; % normalization must be changed
        realization=100*exp(log(DataMACRO(T1av+1:Tend,indmacro(ii))))/normalization;
        
    elseif ii==1;
        aux=squeeze(IRFAsorted(ii,:,round(qqq*M)));
        realization=100*log(DataMACRO(T1av+1:Tend,indmacro(ii)));
    end
    subplot(7,1,count); quantilePlot([2019+.5/12:1/12:2022+5.5/12]', aux,ColorPlot); grid on; hold on    
    
    xlim([2019+.5/12 2022+5.5/12])
    xticks([2019+.5/12 2020+.5/12 2021+.5/12 2022+.5/12])
    xticklabels({'Jan 2019','Jan 2020','Jan 2021','Jan 2022'})
    
    ylabel(YLABELfcst(ii));
    title(series(ii));
end
sgtitle('Covid volatility - est. sample ends in 2021:5','FontSize',12,'FontWeight','bold','color',ColorPlot)


clear all
f42=figure(42);
f42.Position = [0, 0, 350, 800];

load CVFeb2020_May2021
ColorPlot=ColorBase;
count=0;
for ii=1:n
    count=count+1;
    if ii~=1;
        aux=exp(squeeze(IRFAsorted(ii,:,round(qqq*M)))/100); normalization=aux(13,3); aux=100*aux/normalization; % normalization must be changed
        realization=100*exp(log(DataMACRO(T1av+1:Tend,indmacro(ii))))/normalization;
        
    elseif ii==1;
        aux=squeeze(IRFAsorted(ii,:,round(qqq*M)));
        realization=100*log(DataMACRO(T1av+1:Tend,indmacro(ii)));
    end
    subplot(7,1,count); quantilePlot([2019+.5/12:1/12:2022+5.5/12]', aux,ColorPlot); grid on; hold on    
    
    xlim([2019+.5/12 2022+5.5/12])
    xticks([2019+.5/12 2020+.5/12 2021+.5/12 2022+.5/12])
    xticklabels({'Jan 2019','Jan 2020','Jan 2021','Jan 2022'})
    
    ylabel(YLABELfcst(ii));
    title(series(ii));
end
sgtitle('Constant volatility - est. sample ends in 2020:2','FontSize',12,'FontWeight','bold','color',ColorPlot)

for jj=1:n
    figure(41); subplot(7,1,jj); yl=get(gca,'ylim'); YMIN=yl(1); YMAX=yl(2);
    figure(42); subplot(7,1,jj); yl=get(gca,'ylim'); YMIN=[YMIN  yl(1)]; YMAX=[YMAX yl(2)];
    YLIM=[min(YMIN) max(YMAX)];
    figure(41); subplot(7,1,jj); set(gca,'ylim',YLIM); 
    figure(42); subplot(7,1,jj); set(gca,'ylim',YLIM); 
    clear YMIN YMAX YLIM
end