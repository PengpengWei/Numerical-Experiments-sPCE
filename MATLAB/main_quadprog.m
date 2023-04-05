%% Initialization:
%
% Set global parameters in script_env.m
%
script_env

options = optimoptions('quadprog','Display','none');

% Set objective function as \sum_gamma (prob(gamma) - 0.5)^2, to encourage
% mixed strategies.
weights = [1; 1; 1; 1; ...
    1; 1; 1; 1; ...
    1; 1; 1; 1; ...
    1; 1; 1; 1];
H = 2 * eye(16) .* weights;
f = - ones(16, 1) .* weights;

%% Stage 2

rng(315); % Fix the random seed for reproducibility.

phi2 = NaN(size(pi2_vals, 1), 16);
for pi2_ind = 1 : size(pi2_vals, 1)
    A_t2 = A_gen(2, pi2_ind, x_vals, pi2_vals);
    b_t2 = zeros(size(A_t2, 1), 1);
    Aeq_t2 = ones(1, 16);
    beq_t2 = 1;
    lb_t2 = zeros(16, 1); ub_t2 = ones(16, 1);
    distb = (1 + (rand(16, 1) - 0.5) * 0.6); 
    % distb = ones(16, 1);
    sol = quadprog(H .* distb, f .* distb, -A_t2, -b_t2, Aeq_t2, beq_t2, lb_t2, ub_t2, [], options);
    assert(all(A_t2 * sol >= -1e-8));
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
    A_t1 = A_gen(1, pi1_ind, x_vals, pi2_vals, V12, V22);
    b_t1 = zeros(size(A_t1, 1), 1);
    Aeq_t1 = ones(1, 16);
    beq_t1 = 1;
    lb_t1 = zeros(16, 1); ub_t1 = ones(16, 1);
    sol = quadprog(H, f, -A_t1, -b_t1, Aeq_t1, beq_t1, lb_t1, ub_t1, [], options);
    if isempty(sol)
        failed = [failed, pi1_ind];
        continue
    end
    assert(all(A_t1 * sol >= -1e-8));
    phi1(pi1_ind, :) = sol';
end

disp("Stage 1: Done.. The following initial q values have no feasible phi1:")
disp(num2str(q_vals(failed)));
disp(["% of failures: ", num2str(length(failed) / length(q_vals) * 100), ...
    ", min:", num2str(min(q_vals(failed))), ...
    ", max:", num2str(max(q_vals(failed)))]);