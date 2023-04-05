%% Initialization:
%
% Set global parameters in script_env.m
%
script_env

options = optimoptions('linprog','Display','none');


%% Stage 2

rng(12); % Fix the random seed for reproducibility.

phi2 = NaN(size(pi2_vals, 1), 16);
for pi2_ind = 1 : size(pi2_vals, 1)
    A_t2 = [];
    b_t2 = [];
    Aeq_t2 = ones(1, 16);
    beq_t2 = 1;
    lb_t2 = zeros(16, 1); ub_t2 = ones(16, 1);
    f = f_gen_social(2, pi2_ind, x_vals, pi2_vals);
    sol = linprog(-f, -A_t2, -b_t2, Aeq_t2, beq_t2, lb_t2, ub_t2, options);
    % assert(all(A_t2 * sol >= -1e-8));
    phi2(pi2_ind, :) = sol';
end

%% Evaluate the V^i_2

[V12, V22] = V2i_gen(x_vals, pi2_vals, phi2);
disp("Stage 2: Done..")
pause(1);

%% Stage 1

% rng(315); % Fix the random seed for reproducibility.

phi1 = NaN(length(q_vals), 16);
failed = [];
for pi1_ind = 1 : length(q_vals)
    A_t1 = [];
    b_t1 = [];
    Aeq_t1 = ones(1, 16);
    beq_t1 = 1;
    lb_t1 = zeros(16, 1); ub_t1 = ones(16, 1);
    f = f_gen_social(1, pi1_ind, x_vals, pi2_vals, V12, V22);
    sol = linprog(-f, -A_t1, -b_t1, Aeq_t1, beq_t1, lb_t1, ub_t1, options);
    if isempty(sol)
        failed = [failed, pi1_ind];
        continue
    end
    % assert(all(A_t1 * sol >= -1e-8));
    phi1(pi1_ind, :) = sol';
end

disp("Stage 1: Done.. The following initial q values have no feasible phi1:")
disp(num2str(q_vals(failed)));
disp(["% of failures: ", num2str(length(failed) / length(q_vals) * 100), ...
    ", min:", num2str(min(q_vals(failed))), ...
    ", max:", num2str(max(q_vals(failed)))]);

%% Evaluate the expected social welfare
sw2 = expected_sw(phi2, x_vals, pi2_vals);
sw1 = expected_sw(phi1, x_vals, pi2_vals, V12, V22);