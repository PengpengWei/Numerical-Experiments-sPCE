%% Set Parameters Here
q_vals = 0.2;
x_vals = [0.2, 1.2];

%%
% Given q, pi1, pi2 take values from [0,0; 0,q; q,0; 1,q; q,1; 0,1; 1,0;
%  1,1];
% So if aggregate all q's, we have [(0,q); (q,0);
% (q,q); (1,q); (q,1); 0,0; 0,1; 1,0; 1,1].
pi2_vals = ...
    [[q_vals', q_vals']; ...
     [zeros(length(q_vals), 1), q_vals']; ...
     [q_vals', zeros(length(q_vals), 1)]; ...
     [ones(length(q_vals), 1), q_vals']; ...
     [q_vals', ones(length(q_vals), 1)]; ...
     [0, 0; 0, 1; 1, 0; 1, 1];];

