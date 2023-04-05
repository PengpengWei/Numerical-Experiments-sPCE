function [V1, V2] = V2i_gen(x_vals, pi2_vals, phi2)
%V2i_gen.m V^i_2 generator. Generate the V^i_2 functions for stage 2 given 
% possible x values, possible pi2 vectors, and phi2.
% 
% Inputs:
% x_val: the costs corresponding to x^L, x^H.
% pi2_vals: each row is a possible pi2 value.
% phi2: a length(pi2_vals) by 16 matrix, each row represents 
%  a \psi = \hat{\phi}^C_2[\pi_2].
% 
%
% Outputs:
% V1: V^1_2(pi2_ind, x^i_2): a length(pi2_vals) by 2 matrix,
% V2: V^2_2(pi2_ind, x^i_2): a length(pi2_vals) by 2 matrix.
%
%
    gamma_ind = 1 : 16;
    gamma1_ind = fix((gamma_ind - 1) ./ 4) + 1;
    gamma2_ind = mod(gamma_ind, 4) + 4 * (mod(gamma_ind, 4) == 0);
    gamma_ind_switch = (gamma2_ind - 1) * 4 + gamma1_ind;
    
    q_num = (size(pi2_vals, 1) - 4) / 5;
    pi2_ind_switch = ...
        [1 : q_num,...
        2 * q_num + 1 : 3 * q_num, q_num + 1 : 2 * q_num, ...
        4 * q_num + 1 : 5 * q_num, 3 * q_num + 1 : 4 * q_num, ...
        5 * q_num + [1, 3, 2, 4]];
    
    V1 = V21_gen(x_vals, pi2_vals, phi2);
    V2 = V21_gen(x_vals, pi2_vals, phi2(pi2_ind_switch, gamma_ind_switch));
    % Note that we don't have to adjust the order of V2 in the belief axis,
    % because in A_gen we will treat the player 2 as if she were player 1,
    % so we leave the order as-is.
end

function Vi = V21_gen(x_vals, pi2_vals, phi2)
% V21_gen: V^1_2 generator
%
% Generate the V^1_2 function for stage 2.
% 
% Inputs:
% x_val: the costs corresponding to x^L, x^H.
% pi2_vals: each row is a possible pi2 value.
% phi2: a length(pi2_vals) by 16 matrix, each row represents a \psi = \hat{\phi}^C_2[\pi_2].
% 
%
% Outputs:
% Vi: V^1_2(pi2_ind, x^i_2): a length(pi2_vals) by 2 matrix.
%
% Memo:
% 
% Always assume i=1!
%
% x^i_2 = 1, 2
%
    Vi = zeros(size(pi2_vals, 1), length(x_vals));
    for pi2_ind = 1 : size(pi2_vals, 1)
        pi2 = pi2_vals(pi2_ind, :);
        for xi_ind = 1 : length(x_vals)
            for gamma_prof = 1 : 16
                gamma1 = fix((gamma_prof - 1) / 4) + 1;
                gamma2 = mod(gamma_prof, 4) + 4 * (mod(gamma_prof, 4) == 0);
                a1 = (bitand(gamma1 - 1, 3 - xi_ind) ~= 0) + 1;
                a2L = (bitand(gamma2 - 1, 3 - 1) ~= 0) + 1;
                a2H = (bitand(gamma2 - 1, 3 - 2) ~= 0) + 1;
                
                Vi(pi2_ind, xi_ind) = Vi(pi2_ind, xi_ind) + ...
                    (a1 == 2) * (1 - x_vals(xi_ind))  * ...
                     phi2(pi2_ind, gamma_prof) + ... % if a^1_2 = 2, earn 1-x^1
                    (a1 == 1) * ... % if a^1_2 = 1, find the probability of a^2_2=2
                    ((a2L == 2) * (1 - pi2(2)) + ... % case 1: gamma^2 = gamma10 or gamma11, and x^2=L
                     (a2H == 2) * pi2(2)) * ... % case 2: gamma^2 = gamma01 or gamma 11, and x^2=H
                     phi2(pi2_ind, gamma_prof); 
            end
        end
    end
end