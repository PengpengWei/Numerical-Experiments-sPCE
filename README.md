# Numerical Experiments for Coordination in Markov Games with Asymmetric Information

## Game Settings

- 2 players. 2 stages.
- Each player observes a state $x^i \in \{1, 2\}$, determining the cost of contribution:
  - $v(1)$ is the cost for state 1, $v(2)$ is the cost for state 2. $0 < v(1) < 1 < v(2)$.
- With probability $q$, player $i$ observes $x^i=2$.
  - Therefore, the initial common belief can be represented by $\pi_1=[q, q]$.
- At each time $t$, if player $i$ contributes, she receives $1-v(x^i)$ dollars. If player $i$ does not contributes, she receives $1$ dollar if the other player contributes, and receives $0$ dollar if the other player does not contribute. 
  - $a^i_t \in \{1, 2\}$, where 1 represents "not contribute", and "2" represents "contribute".

## Python Code

The reader may refer to the `sPCE.py` directly. The class `PublicInvestProblem` includes the detailed comments for each module. `test.ipynb` presents some sample codes.  

## MATLAB Functions

- `script_env.m`: a script for initialization. It creates all the required global parameters.
  - `x_vals`: $[v(1), v(2)]$
  - `pi2_vals`: all the possible values of $\pi_2$. Note that $\pi_1$'s values are also included.
- `V2i_gen.m`: a function that generates $V^1_2, V^2_2$, given the global vars `x_vals, pi2_vals`, and a given $\hat{\phi}^C_2$. 
  - The output consists of two matrices. Each matrix is indexed by the state $x^i$ and the index of $\pi_2$ in `pi2_vals`. 
  - NOTE THAT both value functions treat the player as the player 1, so for `V2`, if you want to find the value under certain $[\pi^1_2, \pi^2_2]$, the inputted belief index should be the one of $[\pi^2_2, \pi^1_2]$ instead of $[\pi^1_2, \pi^2_2]$.
- `next_belief.m`: a function that finds the index of $\pi_2$
  - Inputs: the index of $\pi_1$, index of $\gamma^1_1, \gamma^2_1, a^1_1, a^2_1$, and the global var `pi2_vals`.
  - $\gamma^i_t$ is indexed by 1, 2, 3, 4, corrresponding to $\gamma_{00}, \gamma_{01}, \gamma_{10}, \gamma_{11}$.
  - BTW, $\gamma_t$ as a prescription profile, is indexed by $1, \ldots, 16$, corresponding to $(\gamma_{00}, \gamma_{00}), (\gamma_{00}, \gamma_{01}), \ldots, (\gamma_{01}, \gamma_{00}), \ldots, (\gamma_{11}, \gamma_{10}), (\gamma_{11}, \gamma_{11})$.
- `A_gen.m`: a function that generates the coefficient matrix $A$ for the rationality constraints.
  - Inputs: time $t$, the index of the current belief $\pi_t$, and global vars `x_vals, pi2_vals`. If $t=1$, it requires the pre-calculated value functions `V1, V2`.
  - Output: the coefficient matrix for the rationality of $\psi_t = \hat{\phi}^C_t[\pi_t]$.
- `f_gen.m`: a function that generates the coefficient matrix $f$ for the calculation of social welfare.
  - Inputs: time $t$, the index of the current belief $\pi_t$, and global vars `x_vals, pi2_vals`. If $t=1$, it requires the pre-calculated value functions `V1, V2`.
  - Output: a vector $f$. $f^T \cdot \hat{\phi}^C_t[\pi_t]$ is the expected social welfare.
- `expected_sw.m`: a function that generates that evaluates the expected social welfare (to go) of a given device.
  - Inputs: device $\hat{\phi}^C_t$, global vars `x_vals, pi2_vals`. The precalculated value functions from last stage `V1,V2`.
  - Output: a social welfare vector. Each row corresponds to an initial belief.
  
- `main_linprog.m`: a main file for the experiment. Use linear programming solver to find a correlation device.
- `main_social.m`: a main file for the experiment. Use linear programming solver to find a correlation device, whose objective function is set to be the expected social welfare to go.
- `main_quadprog.m`: a main file for the experiment. Use quadratic programming solver to find a correlation device.
- `main_team.m`: a main file for the experiment. Use linear programming to solve the team problem.
- `main_test.m`: for test purpose. May be deleted in later versions.

## Experiment Settings

- $v(1) = 0.2$, $v(2) = 1.2$.
- Consider $q = 0.1 : 0.01 : 0.95$.

## Steps

- Set global parameters in `script_env.m`
- Run `main_linprog.m`, `main_social.m` or `main_quadprog.m`:
  - Use `script_env.m` for initialization.
  - Choose a random seed for reproducibility of the experiment results.
  - Find a correlation device for Stage 2. For all the possible $\pi_2$ values, find a $\psi_2 = \hat{\phi}^C_2[\pi_2]$:
    - Generate the coefficient matrix $A$ by `A_gen.m`
    - Generate an objective function for linear programming or quadratic programming.
    - Find a solution of the linear/quadratic programming, subject to the constraints given by $A$, and the fact that $\psi_t$ is a probability distribution. 
    - Save the result in the matrix `phi2`.
  - Evaluate the value functions using `V2i_gen.m`
  - Find a correlation device for Stage 1. 
    - All the steps are similar to those in Stage 2.
    - Note that the $\pi_1$ values are the first `length(q_vals)` rows in `pi2_vals`.
    - It is possible that for some $\pi_1$ the solution does not exist. Store those indices of failure in an array.

## Useful Facts

- Belief update: suppose we know $\pi^i_1$,
  - If $\gamma^i_1 = \gamma_{00}, \gamma_{11}$, then $\pi^i_2 = \pi^i_1$.
  - If $\gamma^i_1 = \gamma_{01}$ or $\gamma_{10}$, from $a^i_1$ one may infer the exact $x^i$, so $\pi^i_2 = 0$ or $1$.
- Adjust the order of gamma profile vector when players 1 and 2 switch:
  - For each $\gamma$, the $index-1$ is exactly the decimal form of the subscript of $\gamma^1, \gamma^2$. For example, $\gamma$-index=5 implies the subscripts of the profile is $4 = (0100)_2$, so it corresponds to $(\gamma_{01}, \gamma_{00})$.
  - To switch the order:
    - The binary form of $\gamma^1$ is fix(($\gamma$-index - 1) / 4), so the MATLAB index is `gamma1_ind = fix((gamma_ind - 1) ./ 4) + 1;`
    - The binary form of $\gamma^2$ is mod(($\gamma$-index - 1), 4), so the MATLAB index is `gamma2_ind = mod(gamma_ind, 4) + 4 * (mod(gamma_ind, 4) == 0);`
    - The switched $\gamma$-index is  `(gamma2_ind - 1) * 4 + gamma1_ind;`
- Given the index of $\gamma^i \in \{1,2,3,4\}$ and $x^i \in \{1,2\}$, how do we calculate $a^i \in \{1, 2\}$?
  - The index of $\gamma^i$ minus 1 corresponds to its binary form 00, 01, 10, 11.
  - The first bit implies the action at $x^i=1$, and the second bit implies the action at $x^i=2$.
  - Notice that $3-x^i$'s binary form is 10 and 01.
  - Therefore, if we do `bitand(gammai - 1, 3 - xi)`, then we will get 0 if "not contribute" is taken, get 2 or 1 if "contribute" is taken. 