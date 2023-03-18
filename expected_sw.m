function sw = expected_sw(phi, x_vals, pi2_vals, V1, V2)
%expected_sw.m: Expected social welfare (to go)
% Calculate the expected social welfare of the given correlation device. 
% 
% Inputs:
% phi: the correlation device at current time t. 
% x_val: a 2-dim vector indicating the cost at xL and xH.
% pi2_vals: each row is a possible pi2 value.
% V1: value function of player 1 from the next stage 
% V2: value function of player 2 from the next stage 
%
% Outputs:
% sw: expected social welfare (to go). A size(phi, 1) by 1 vector, each
%  entry represents the expected social welfare given the corresponding
%  public belief.
% 

    f_mat = zeros(16, size(phi, 1));
    
    for pi_ind = 1 : size(phi, 1)
        if nargin == 3
            f_mat(:, pi_ind) = f_gen_social(2, pi_ind, x_vals, pi2_vals);
        else
            f_mat(:, pi_ind) = f_gen_social(1, pi_ind, x_vals, pi2_vals, V1, V2);
        end
    end
    
    sw = sum(f_mat' .* phi, 2);
end