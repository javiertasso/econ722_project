% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% Simulations change point models - first try 
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

clearvars;
clc; 
addpath([cd '\functions'])
figures_dir = 'C:\Users\Jota\Dropbox\Aplicaciones\Overleaf\Econ722-Term-Paper\';

% Simulate some data
T = 200; 
mmu_true = 0;
time = transpose(1:T); 

% Frequentist analysis ----------------------------------------------------
    % Single changepoint 
    % H0 = No changepoint
    % H1 = There is a changepoint

N_sim = 10000; 
reject = zeros(N_sim, 1); 
change_point_estimate = NaN(N_sim,1); 
possible_sigma = [1.25; 1.5; 2; 4]; 

for kk = 1:4 

    ssigma2 = possible_sigma(kk); 

    for ii = 1:N_sim 
    
        data = sim_data_var(T, mmu_true, 1, ssigma2); 
        % Under the null assume there is no change point 
        ssigma_2_hat_null = transpose(data)*data / (T); 
        log_like_null = sum(log(normpdf(data, mmu_true, sqrt(ssigma_2_hat_null)))); 
        
        % Under the alternative calculate the likelihood of a changepoint on each
        % time from 2 to T-2, remember I need at least 2 points to estimate the
        % variance 
        variance_1_alternative = NaN(T,1); 
        variance_2_alternative = NaN(T,1); 
        log_like_alt_for_each_t = NaN(T,1); 
        
        for tt = 2:(T-2)
        
            variance_1_alternative(tt,1) = transpose(data(1:tt)) * data(1:tt) / (tt);
            variance_2_alternative(tt,1) = transpose(data((tt+1):T)) * data((tt+1):T) / (T-tt);
            log_like_alt_for_each_t(tt,1) = sum(log(normpdf(data(1:tt), mmu_true, sqrt(variance_1_alternative(tt,1))))) + ...
                sum(log(normpdf(data((tt+1):T), mmu_true, sqrt(variance_2_alternative(tt,1))))); 
        
        end
        
        % Likelihood ratio test 
        llambda = sqrt(max(log_like_alt_for_each_t) - log_like_null); 
        [~, change_point_estimate(ii,1)] = max(log_like_alt_for_each_t); 
        critical_value = (fzero(@(x) exp(-2*exp(-x)) - 0.95, 4)); 
        a_temp = sqrt(2*log(log(T))); 
        b_temp = 2*log(log(T)) + 1/2 * log(log(log(T))) - log(gamma(1/2)); 
        llambda_modified = a_temp * llambda - b_temp; 
    
        if llambda_modified > sqrt(critical_value)
    
            reject(ii,1) = 1; 
             
        end
    
    end
    
    figure(1)
    histogram(change_point_estimate(reject==1))
    hold on
    line([100, 100], ylim, 'LineWidth', 2, 'Color', 'r');
    xlabel('Change Point')
    ylabel('Frequency')
    title(['Estimated Time of the Change for \sigma= ', num2str(ssigma2)])
    subtitle(['Detected ', num2str(sum(reject)), ' of ', num2str(N_sim)])
    hold off
    saveas(gcf,[figures_dir sprintf('change_point_freq_%d.png',kk)]);
    close(figure(1))

end
