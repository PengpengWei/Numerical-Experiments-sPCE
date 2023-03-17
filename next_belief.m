function next_pi_ind = next_belief(pi_ind, pi2_vals, gamma1, gamma2, a1, a2)
%next_belief.m Find the index of the next belief.
% 
% Inputs:
%
% pi_ind: current belief index.
% pi2_vals: all the possible pi2 values.
% gamma1: player1's prescription.
% gamma2: player2's prescription.
% a1: player1's action.
% a2: player2's action.
%
% Outputs:
%
% next_pi_ind: the index of the next belief.
%
% Memo:
% Current belief: pi1 = [q, q];
%
%

    pi_current = pi2_vals(pi_ind, :);
    if gamma1 == 1 || gamma1 == 4
        pi_next(1) = pi_current(1); % remains unchanged: =q
    elseif gamma1 == 2 % gamma1 = gamma01
        if a1 == 1 % not contribute
            pi_next(1) = 0; % sure that xi = L
        else
            pi_next(1) = 1;
        end
    else % gamma1 = gamma10
        if a1 == 1 % not contribute
            pi_next(1) = 1; % sure that xi = H
        else
            pi_next(1) = 0;
        end
    end

    if gamma2 == 1 || gamma2 == 4
        pi_next(2) = pi_current(2); % remains unchanged: =q
    elseif gamma2 == 2 % gamma2 = gamma01
        if a2 == 1 % not contribute
            pi_next(2) = 0; % sure that xi = L
        else
            pi_next(2) = 1;
        end
    else % gamma2 = gamma10
        if a2 == 1 % not contribute
            pi_next(2) = 1; % sure that xi = H
        else
            pi_next(2) = 0;
        end
    end

    next_pi_ind = find(ismember(pi2_vals, pi_next, 'row'));
end