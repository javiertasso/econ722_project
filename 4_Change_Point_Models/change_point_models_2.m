% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% Simulations change point models 2 - first try 
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

clearvars;
clc; 
addpath([cd '\functions'])
figures_dir = 'C:\Users\Jota\Dropbox\Aplicaciones\Overleaf\Econ722-Term-Paper\';
run_old_stuff = 0; 
rng(1234)

% Simulate some data
T = 200; 
mmu_true = 0;
time = transpose(1:T); 
ssigma2 = 4; 
data = sim_data_var(T, mmu_true, 1, ssigma2);

% Values for hyperparameters
ttheta = 0.01; 
v = 1;
c = (v+1) / 2; 
% llambda = 1 / T; 

% Try the function 
[post_odds, post_q_given_x, post_k_given_xq] = ...
    bayesian_change_point(100000, data, ttheta, v, c, 4); 

disp(post_odds(2,3))
disp(post_odds(3,2))
disp(post_q_given_x(2))
disp(post_k_given_xq(100, :, 2))
clc 

% Given that there was one change, when did it happen? 
post_distrib_time = post_k_given_xq(:,1:2, 2); 
[sort_k, index_temp] = sort(post_distrib_time(:,2)); 
sort_post = post_distrib_time(index_temp,1); 
clear index_temp

% Make figure
figure(1)
plot(sort_k, sort_post, 'LineWidth',2)
hold on 
set(gca,'Ytick',[])
xlabel('Moment when Change Happened')
ylabel('Density')
xlim([90,110])
line([100, 100], ylim, 'LineWidth', 2, 'Color', 'r');
title('$f(k|y, \hat{q}, \sigma=4)$', 'Interpreter','Latex');
subtitle(['Odds Ratio 1 over 2: ', num2str(round(post_odds(2,3),2))])
saveas(gcf,[figures_dir 'change_point_bayes_0.png']);
close(figure(1))
 

writematrix(round(post_odds,4),...
    [figures_dir 'change_point_bayes_odds_0.csv']); 
% saveas(gcf,[figures_dir sprintf('change_point_freq_%d.png',kk)]);

% Now loop over different scenarios 
possible_sigma = [1.25; 1.5; 2; 4]; 

for kk = 1:4 

    ssigma2 = possible_sigma(kk); 

    % Simulate some data
    data = sim_data_var(T, mmu_true, 1, ssigma2);

    % Try the function 
    [post_odds, ~, post_k_given_xq] = ...
        bayesian_change_point(100000, data, ttheta, v, c, 4); 

    % Given that there was one change, when did it happen? 
    post_distrib_time = post_k_given_xq(:,1:2, 2); 
    [sort_k, index_temp] = sort(post_distrib_time(:,2)); 
    sort_post = post_distrib_time(index_temp,1); 
    clear index_temp
    
    % Make figure
    figure(1)
    plot(sort_k, sort_post, 'LineWidth',2)
    hold on 
    set(gca,'Ytick',[])
    xlabel('Moment when Change Happened')
    ylabel('Density')
    xlim([50,150])
    line([100, 100], ylim, 'LineWidth', 2, 'Color', 'r');
    title('$f(k|y, \hat{q})$', 'Interpreter','Latex');
    subtitle(['Odds Ratio 1 over 2: ', num2str(round(post_odds(2,3),2))])
    saveas(gcf,[figures_dir sprintf('change_point_bayes_%d.png',kk)]);
    close(figure(1))
     
    
    
    writematrix(round(post_odds,4),...
        [figures_dir sprintf('change_point_bayes_odds_%d.csv', kk)]); 
    % saveas(gcf,[figures_dir sprintf('change_point_freq_%d.png',kk)]);


end

 


%%

if run_old_stuff == 1 

% Total number of change points
% q_draw = binornd(T-1, ttheta); 
%q_draw = 1; 
q_possible = [0;1;2;3;4]; 
post_q_given_x = NaN(5,1); 

for kk = 1:5 

    q_draw = q_possible(kk); 

    % Draw from prior distributions -------------------------------------------
    
    % Draws for the precision parameter (inverse of the variance)
    rand_temp = rand(q_draw + 1, 1); 
    eta_draw = gaminv(rand_temp, v, c); 
    clear rand_temp 
    
    % Set up algorithm
    N_sim = 100000; 
    joint_xk_given_q = NaN(N_sim,1); 
    k_draws_store = NaN(q_draw + 2, N_sim); 
    
    
    for ii = 1:N_sim
       
        % Draw for vector k = (k1, k2, ..., kq) that tells me where the regime
        % changes
        k_draw = NaN(q_draw + 2, 1); 
        k_draw(1,1) = 0;
        k_draw(q_draw + 2, 1) = T; 
        k_draw(2:(q_draw+1),1) = time(sort(randperm(196,q_draw) + 2)); 
        
        % Vector of standard deviations 
        sd_vector = sqrt(1/eta_draw(1)) * ones(T,1); 
        
        if q_draw > 0 
        
            for qq = 1:q_draw
        
                sd_vector(k_draw(qq+1):k_draw(qq+2), 1) = sqrt(1/eta_draw(qq+1));  
        
            end
        
        end
        
        
        % Create vectro dj
        dj_vector = NaN(q_draw + 1, 1); 
        
        for qq = 1:(q_draw+1)
        
            dj_vector(qq) = k_draw(qq+1) - k_draw(qq); 
        
        end 
        
        
        % Given (eta, k, q) calculate the likelihood (Normal distribution)
        log_likelihood = sum(log(normpdf(data, mmu_true, sd_vector))); 
        
        % Given q, calculate the (log) joint density of (x, eta, k)
        log_joint_density = log(1 / nchoosek(T-1, q_draw)) ...
            + sum(log((c^v / gamma(v)) .* eta_draw.^(-v-1) .* exp(-c ./eta_draw))) ...
            + log_likelihood; 
        
        % Integrate out the etas
        
            % Temporal vector to add up data 
            temp_data_squared = zeros(200, q_draw + 1); 
        
            for qq = 1:(q_draw + 1)
        
                temp_data_squared(k_draw(qq)+1:k_draw(qq+1),qq) = ...
                    data(k_draw(qq)+1:k_draw(qq+1),1).^2; 
        
            end
            
            sum_temp_data_squared = transpose(sum(temp_data_squared)); 
            clear temp_data_squared 
        
            temp_sss = sum(-((dj_vector+v)./2) .* log(1/2 * (sum_temp_data_squared) + c) + ...
            log(gamma((dj_vector + v)/2)));
             
        % This is the log likelihood equation (2.28) 
        log_joint_xk_given_q = log( 1 / ((2*pi)^(T/2) * nchoosek(T-1,q_draw)) * ...
            (c^(v/2) / gamma(v/2))^(q_draw + 1)) + temp_sss; 
        joint_xk_given_q(ii) = exp(log_joint_xk_given_q); 
    
        % Store K_draws
        k_draws_store(:,ii) = k_draw; 
        
        clear temp_sss sum_temp_data_squared 
    
    end
    
    temporal_matrix = [transpose(k_draws_store(2:q_draw+1,:)), joint_xk_given_q]; 
    temporal_matrix = [temporal_matrix, joint_xk_given_q / sum(joint_xk_given_q)]; 

   

     
    
    if q_draw == 1
        
        stop 
        [K_sorted, idx] = sort(temporal_matrix(:,1));
        post_sorted = temporal_matrix(idx,3); 
        figure(1)
        plot(K_sorted, post_sorted)
        close(figure(1))
        clear idx post_sorted K_sorted
    
    end
    
    density_x_given_q = mean(joint_xk_given_q); 
    post_q_given_x(kk) = binopdf(q_draw, T-1, ttheta) * density_x_given_q; 

end

% Odds ratio 
odds12 = post_q_given_x(2) / post_q_given_x(3);
odds10 = post_q_given_x(2) / post_q_given_x(1);

disp(odds12)
disp(odds10)

end 


quit 




