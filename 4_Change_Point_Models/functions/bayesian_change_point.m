function [post_odds, post_q_given_x, post_k_given_xq] = ...
    bayesian_change_point(N_sim, data, ttheta, v, c, q_max)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here



% Total number of change points
q_possible = transpose(0:q_max); 
post_q_given_x = NaN(length(q_possible),1); 
T = length(data); 
time = transpose(1:T);
post_k_given_xq = zeros(N_sim, length(q_possible), length(q_possible)); 

for kk = 1:(length(q_possible))

    q_draw = q_possible(kk); 

    % Draw from prior distributions -------------------------------------------
    
    % Draws for the precision parameter (inverse of the variance)
    rand_temp = rand(q_draw + 1, 1); 
    eta_draw = gaminv(rand_temp, v, c); 
    clear rand_temp 
    
    % Set up algorithm
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
        % log_likelihood = sum(log(normpdf(data, mmu_true, sd_vector))); 
        
        % Given q, calculate the (log) joint density of (x, eta, k)
        % log_joint_density = log(1 / nchoosek(T-1, q_draw)) ...
            % + sum(log((c^v / gamma(v)) .* eta_draw.^(-v-1) .* exp(-c ./eta_draw))) ...
            % + log_likelihood; 
        
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
    
    % temporal_matrix = [transpose(k_draws_store(2:q_draw+1,:)), joint_xk_given_q]; 
    % temporal_matrix = [temporal_matrix, joint_xk_given_q / sum(joint_xk_given_q)]; 
    post_k_given_xq(:,1,kk) = joint_xk_given_q / sum(joint_xk_given_q); 
    post_k_given_xq(:,2:q_draw+1,kk) = transpose(k_draws_store(2:q_draw+1,:)); 
    
    % if q_draw == 1
    
        % [K_sorted, idx] = sort(temporal_matrix(:,1));
        % post_sorted = temporal_matrix(idx,3); 
        % figure(1)
        % plot(K_sorted, post_sorted)
        % close(figure(1))
        % clear idx post_sorted K_sorted
    
    % end
    
    density_x_given_q = mean(joint_xk_given_q); 
    post_q_given_x(kk) = binopdf(q_draw, T-1, ttheta) * density_x_given_q; 

end

% Odds ratio 
% odds12 = post_q_given_x(2) / post_q_given_x(3);
% odds10 = post_q_given_x(2) / post_q_given_x(1);

post_odds = NaN(length(q_possible), length(q_possible)); 

for kk = 1:(length(q_possible))

    post_odds(kk, :) = post_q_given_x(kk) ./ post_q_given_x; 

end


end