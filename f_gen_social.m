function f = f_gen_social(t, pi_ind, x_vals, pi2_vals, V1, V2)
%f_gen_social.m f generator. 
% Generate an f matrix for social welfare.
%
% Inputs:
% t: time, 1 or 2
% pi_ind: current belief index, 1~length(pi2_vals). (Note that pi1's value
%  is also in pi2_vals)
% x_val: a 2-dim vector indicating the cost at xL and xH.
% pi2_vals: each row is a possible pi2 value.
% V1: value function of player 1 from the next stage 
% V2: value function of player 2 from the next stage 
%
% Outputs:
% f matrix: 16 by 1.
%

    pi_value = pi2_vals(pi_ind, :);
    pi_value_switched = [pi_value(2), pi_value(1)];
    pi_ind_switch = find(ismember(pi2_vals, pi_value_switched, 'row'));
    
    if nargin == 4
        f1 = f1_gen(t, pi_ind, x_vals, pi2_vals);
        f2 = f1_gen(t, pi_ind_switch, x_vals, pi2_vals);
    else
        f1 = f1_gen(t, pi_ind, x_vals, pi2_vals, V1);
        f2 = f1_gen(t, pi_ind_switch, x_vals, pi2_vals, V2);
    end
    
    gamma_ind = 1 : 16;
    gamma1_ind = fix((gamma_ind - 1) ./ 4) + 1;
    gamma2_ind = mod(gamma_ind, 4) + 4 * (mod(gamma_ind, 4) == 0);
    gamma_ind_switch = (gamma2_ind - 1) * 4 + gamma1_ind;
    
    f = f1 + f2(gamma_ind_switch);
end

function f1 = f1_gen(t, pi_ind, x_vals, pi2_vals, V)
%f1_gen.m f1 generator. 
% Generate an f1 matrix for social welfare. f = f1 + f2.
%
% Inputs:
% t: time, 1 or 2
% pi_ind: current belief index, 1~length(pi2_vals). (Note that pi1's value
%  is also in pi2_vals)
% x_val: a 2-dim vector indicating the cost at xL and xH.
% pi2_vals: each row is a possible pi2 value.
% V: value function of player 1 from the next stage 
%
% Outputs:
% f1 matrix: 16 by 1.
%
    f1 = zeros(16, 1);
    pi1 = pi2_vals(pi_ind, 1);
    pi2 = pi2_vals(pi_ind, 2);
    for gamma1 = 1 : 4
        for gamma2 = 1 : 4
            gamma_ind = (gamma1 - 1) * 4 + gamma2;
            for x1 = 1 : 2
                a1 = (bitand(gamma1 - 1, 3 - x1) ~= 0) + 1;
                
                for x2 = 1 : 2
                    if x1 == 1
                        prob = (1 - pi1);
                    else 
                        prob = pi1;
                    end
                    
                    if x2 == 1
                        prob = prob * (1 - pi2);
                    else
                        prob = prob * pi2;
                    end
                    
                    a2 = (bitand(gamma2 - 1, 3 - x2) ~= 0) + 1;
                    
                    f1(gamma_ind) = f1(gamma_ind) + ...
                        prob * ...
                        ((a1 == 2) * (1 - x_vals(x1)) + ...
                        (a1 == 1) * (a2 == 2));
                    if t == 1
                        f1(gamma_ind) = f1(gamma_ind) + ...
                            prob * ...
                            V(next_belief(pi_ind, pi2_vals, gamma1, gamma2, a1, a2), x1);
                    end
                end
            end
        end
    end
end