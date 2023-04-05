function A = A_gen(t, pi_ind, x_vals, pi2_vals, V1, V2)
%A_gen.m A generator. 
% Generate an A matrix for rationality check for both players.
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
% A matrix: (2*8) by 16: 2 players, 4 prescription options, 2 possible
% state componenets. 16 Possible prescription profiles.
% 
% Memo:
%
% Constraint: 
% A(:, 1:16) * psi >= 0,
% psi >= 0, 
% sum(psi) = 1.
%
    pi_value = pi2_vals(pi_ind, :);
    pi_value_switched = [pi_value(2), pi_value(1)];
    pi_ind_switch = find(ismember(pi2_vals, pi_value_switched, 'row'));
    
    if nargin == 4
        A1 = A1_gen(t, pi_ind, x_vals, pi2_vals);
        A2 = A1_gen(t, pi_ind_switch, x_vals, pi2_vals);
    else
        A1 = A1_gen(t, pi_ind, x_vals, pi2_vals, V1);
        A2 = A1_gen(t, pi_ind_switch, x_vals, pi2_vals, V2);
    end
    
    gamma_ind = 1 : 16;
    gamma1_ind = fix((gamma_ind - 1) ./ 4) + 1;
    gamma2_ind = mod(gamma_ind, 4) + 4 * (mod(gamma_ind, 4) == 0);
    gamma_ind_switch = (gamma2_ind - 1) * 4 + gamma1_ind;
    
    A2(:, gamma_ind) = A2(:, gamma_ind_switch); 
    
    A = [A1; A2];
end

function A = A1_gen(t, pi_ind, x_vals, pi2_vals, V)
% A1_gen: A1 generator:
% 
% Generate an A matrix for rationality check assuming i = 1
%
% Inputs:
% (NOTE: assuming that the A matrix is for player 1. so please reorder the
% input before invoking this function when checking player 2.)
% t: time, 1 or 2
% pi_ind: current belief index, 1~length(pi2_vals). (Note that pi1's value
%  is also in pi2_vals)
% x_val: a 2-dim vector indicating the cost at xL and xH.
% pi2_vals: each row is a possible pi2 value.
% V: value function from the next stage
% 
%
% Outputs:
% 
% "A" matrix: 8 by 16.
%
    if nargin == 4
        V = 0;
    end

    A = zeros(8, 16);
    sig_vec = ones(8, 1);

    pi_t = pi2_vals(pi_ind, :);

    for xi = 1 : 2
        for gammai = 1 : 4
            row_no = (xi - 1) * 4 + gammai;

            % Note: if prob of gammai = 0, the values below do not matter.

            % Constant terms.
            for gamma2 = 1 : 4
                A(row_no, (gammai - 1) * 4 + gamma2) = 1 - x_vals(xi);
            end

            % case: x^-i = L; player 2 plays "contribute" only if gamma^-i
            % = gamma10 or 11
            A(row_no, (gammai - 1) * 4 + 3) = A(row_no, (gammai - 1) * 4 + 3) - (1 - pi_t(2)); % gamma^-i = gamma10
            A(row_no, (gammai - 1) * 4 + 4) = A(row_no, (gammai - 1) * 4 + 4) - (1 - pi_t(2)); % gamma^-i = gamma11
            % case: x^-i = H; player 2 plays "contribute" only if gamma^-i
            % = gamma01 or 11
            A(row_no, (gammai - 1) * 4 + 2) = A(row_no, (gammai - 1) * 4 + 2) - pi_t(2); % gamma^-i = gamma01
            A(row_no, (gammai - 1) * 4 + 4) = A(row_no, (gammai - 1) * 4 + 4) - pi_t(2); % gamma^-i = gamma11

            if t == 1
                for gamma2 = 1 : 4
                    a2L = 1 + (gamma2 > 2); % a2L=2 only if gamma2 == gamma10 (3) or gamma11 (4)
                    a2H = 1 + (gamma2 == 2 || gamma2 == 4); % a2H=2 only if gamma2 == gamma01 or gamma11
                    A(row_no, (gammai - 1) * 4 + gamma2) = A(row_no, (gammai - 1) * 4 + gamma2) + ...
                        pi_t(2) * ... % x^-i=H
                        (V(next_belief(pi_ind, pi2_vals, gammai, gamma2, 2, a2H), xi) - ...
                        V(next_belief(pi_ind, pi2_vals, gammai, gamma2, 1, a2H), xi)) + ...
                        (1 - pi_t(2)) * ... % x^-i=L
                        (V(next_belief(pi_ind, pi2_vals, gammai, gamma2, 2, a2L), xi) - ...
                        V(next_belief(pi_ind, pi2_vals, gammai, gamma2, 1, a2L), xi));
                end
            end
            if bitand(gammai - 1, 3 - xi) == 0 % which means playeri plays "not contribute". (note that (gammai-1) is in {00, 01, 10, 11}, and 3 - xi = 2 or 1 = 10 or 01; do bitand while extract the action taken under xi.)
                sig_vec((xi - 1) * 4 + gammai) = -1;
            end
        end
    end

    A = A .* sig_vec;
end