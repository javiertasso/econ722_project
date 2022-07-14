function [log_likelihood, regime_cond, regime_0_likeli_cond, ...
    regime_1_likeli_cond, likeli_vector, regime_smooth, ...
    q0tt_smooth, q1tt_smooth, st_given_next] = ...
    filter_example(aalpha0, aalpha1, ...
    q00, q11, ssigma, q00_initial, q11_initial, data_ts, scale_dev)
% filter_example is the filter for an example to get started
%   Inputs: parameters and initial values
%   Outputs: likelihood function and st/Y(1:t)
%       All outputs with _cond mean they are conditional up to the
%       information at t, that is, before doing the smoothing
%       The other outputs correspond to the case after smoothing


    % Set places to initialize things
    T = length(data_ts);
    q0tt = zeros(T,1);
    q1tt = zeros(T,1); 
    likeli_vector = zeros(T,1); 
    regime_cond = zeros(T,1); 
    
    if not(scale_dev == 1)

        aalpha1 = 0;

    end

    
    % First step is unusual 
        % Forecasting q0 and q1 given initial values
        q0_fore = q00_initial * q00 + q11_initial * (1-q11); 
        q1_fore = q00_initial * (1-q00) + q11_initial * q11; 
    
        % First element of the likelihood 
        likeli_vector(1) = q0_fore * normpdf(data_ts(1), aalpha0, ssigma) + ...
            q1_fore * normpdf(data_ts(1), aalpha0 + aalpha1, scale_dev * ssigma);
    
        % Fill up the qs
        q0tt(1) = (q0_fore * normpdf(data_ts(1), aalpha0, ssigma)) / ...
            likeli_vector(1); 
        q1tt(1) = 1 - q0tt(1);
    
        % Classify regime 
        if q1tt(1) > q0tt(1)
    
            regime_cond(1) = 1;
    
        end
    
    for tt = 2:T
    
        % Forecasting q0 and q1 given y(t-1)
        q0_fore = q0tt(tt-1) * q00 + q1tt(tt-1) * (1-q11); 
        q1_fore = q0tt(tt-1) * (1-q00) + q1tt(tt-1) * q11; 
    
        % Forecasting y(t) given y(t-1)
        y_fore = q0_fore * normpdf(data_ts(tt), aalpha0, ssigma) + ...
            q1_fore * normpdf(data_ts(tt), aalpha0 + aalpha1, scale_dev * ssigma);  
    
        % Fill up the qs
        q0tt(tt) = (q0_fore * normpdf(data_ts(tt), aalpha0, ssigma)) / y_fore; 
        q1tt(tt) = 1 - q0tt(tt);
    
        % Fill up the conditional likelihood for each t
        likeli_vector(tt) = y_fore; 
    
        % Classify regime 
        if q1tt(tt) > q0tt(tt)
    
            regime_cond(tt) = 1;
    
        end
    
    end
    
    log_likelihood = sum(log(likeli_vector)); 
    regime_0_likeli_cond = q0tt; 
    regime_1_likeli_cond = q1tt; 
    
    % Smoothing 
    q0tt_smooth = zeros(T,1);
    q1tt_smooth = zeros(T,1); 
    regime_smooth = zeros(T,1); 
    st_given_next = zeros(T,4);
    
    % Complete the last one 
    q0tt_smooth(T) = q0tt(T);
    q1tt_smooth(T) = q1tt(T);
    regime_smooth(T) = regime_cond(T); 
    st_given_next(T,1) = q0tt(T) * (q0tt(T) * q00 + q1tt(T) * (1-q11));
    st_given_next(T,2) = q1tt(T) * (q0tt(T) * q00 + q1tt(T) * (1-q11));
    st_given_next(T,3) = q0tt(T) * (q0tt(T) * (1-q00) + q1tt(T) * q11);
    st_given_next(T,4) = q1tt(T) * (q0tt(T) * (1-q00) + q1tt(T) * q11);
    
    for tt = 1:(T-1)
    
        st_given_next(T-tt,1) = q0tt(T-tt) * (q0tt(T-tt) * q00 + q1tt(T-tt) * (1-q11));
        st_given_next(T-tt,2) = q1tt(T-tt) * (q0tt(T-tt) * q00 + q1tt(T-tt) * (1-q11));
        st_given_next(T-tt,3) = q0tt(T-tt) * (q0tt(T-tt) * (1-q00) + q1tt(T-tt) * q11);
        st_given_next(T-tt,4) = q1tt(T-tt) * (q0tt(T-tt) * (1-q00) + q1tt(T-tt) * q11);
    
    
    end 
    
    total_temp = st_given_next(:,1) + st_given_next(:,2); 
    st_given_next(:,1) = st_given_next(:,1) ./ total_temp; 
    st_given_next(:,2) = st_given_next(:,2) ./ total_temp;
    total_temp = (st_given_next(:,3) + st_given_next(:,4)); 
    st_given_next(:,3) = st_given_next(:,3) ./ total_temp;
    st_given_next(:,4) = st_given_next(:,4) ./ total_temp;
    clear total_temp 
    
    % Here I need a loop going backwards 
    
    for tt = 1:(T-1)
    
        q0tt_smooth(T-tt) = q0tt_smooth(T-tt+1) * st_given_next(T-tt,1) + ...
            q1tt_smooth(T-tt+1) * st_given_next(T-tt,3); 
        q1tt_smooth(T-tt) = q0tt_smooth(T-tt+1) * st_given_next(T-tt,2) + ...
            q1tt_smooth(T-tt+1) * st_given_next(T-tt,4); 
        
        if q1tt_smooth(T-tt) > q0tt_smooth(T-tt)
    
            regime_smooth(T-tt) = 1;
    
        end
    
    end




end