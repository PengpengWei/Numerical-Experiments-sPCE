import numpy as np
from scipy.optimize import linprog
import time
import pickle
import matplotlib.pyplot as plt


class PublicInvestProblem:
    def __init__(self, q_vals=0.1, x_vals=[0.2, 1.2], num_quant=2, T=2, tol_dig=8, filename=None):
        """Initialize a public investment problem instance. 

        Args:
        ---
        :q_vals: an array of possible values of Q(x^i = H). q_vals = 0.1 by default.

        :x_vals: the costs of contribution corresponding to states xL and xH. x_vals = [0.2, 1.2] by default.

        :num_quant: the number of quantized pts along an axis. num_quant = 2 by default.

        :T: time horizon. T = 2 by default.

        :tol_dig: tolerant digit: set tolerance of belief to be 10^(-tol_dig). tol_dig = 8 by default. Set tol_dig=-1 if not needed.

        :filename: initialize the object using a saved file "filename.pkl". None by default.

        Returns:
        ---
        None.

        """
        if filename: 
            self.load(filename)
            return 
        
        # Read the input vars
        self.q_vals = np.array(q_vals).reshape(-1, ).flatten()
        self.x_vals = np.array(x_vals).reshape(-1, ).flatten()
        self.num_quant = num_quant
        self.T = T
        self.tol_dig = tol_dig

        # Initialize other vars.
        # Instantaneous reward.
        # self.reward = lambda i, x0, x1, a0, a1: \
        #     (a0 + a1 > 0) * 1 - (i == 0 and a0 == 1) * self.x_vals[x0] - \
        #         (i == 1 and a1 == 1) * self.x_vals[x1]

        # A list of possible gamma values, ranging from 0 to 1.
        self.gamma_vals = np.linspace(0, 1, num_quant)

        # A list of possible pi^i values. 
        self.pi_vals = [None] * (self.T + 1) # The set of single pi^i_t values. t = 0, 1, ..., T-1, T (time T is useless though)
        self.__find_belief_candidate() # Construct the pi_vals list.

        # Vectors / Matrices for storing the results.
        # self.phi[t] is the device at time t. Format: self.phi[t][pi0_ind, pi1_ind, gamma0L_ind, gamma0H_ind, gamma1L_ind, gamma1H_ind].
        self.phi = [np.zeros((len(self.pi_vals[t]), len(self.pi_vals[t]), self.num_quant, self.num_quant, \
                              self.num_quant, self.num_quant)) for t in range(self.T)]
        # self.V[t] is the value to go function. Format: self.V[t][i, pi0_ind, pi1_ind, xi] (note that t can be T.)
        self.V = [np.zeros((2, len(self.pi_vals[t]), len(self.pi_vals[t]), 2)) for t in range(T + 1)]
        # self.social_welfare is the social welfare to go. 
        # Format: self.social_welfare[t][pi0_ind, pi1_ind].
        # Exception: index 0: 1-d: each pi_ind represents pi0_ind = pi1_ind = pi_ind.
        self.social_welfare = [np.zeros((len(self.pi_vals[t]), len(self.pi_vals[t]))) for t in range(T + 1)]

        self.__complete = False # A var that indicates if the solver has been run or not.
        self.__team = False # A var that indicates if the solver has been run under team mode or not.
        self.__is_compared = False # A var that indicates if the compare() has been run or not.

        # For sPBE check.
        self.__checked_sPBE = [False] * self.T
        # self.sPBE_flag indicates if the device starts from the current belief is an sPBE. 
        # Format: self.sPBE_flag[t][pi0_ind, pi1_ind].
        # Exception: index 0: 1-d: each pi_ind represents pi0_ind = pi1_ind = pi_ind.
        self.sPBE_flag = [np.full((len(self.pi_vals[t]), len(self.pi_vals[t])), True) for t in range(self.T)]

        # For comparison purpose.
        self.phi_comp = self.phi
        self.V_comp = self.V
        self.social_welfare_comp = self.social_welfare
        self.sPBE_flag_comp = self.sPBE_flag

        return
    
    def reset(self, q_vals=None, x_vals=None, num_quant=None, T=None, tol_dig=None):
        """Reset the environment.

        Args:
        ---
        :q_vals: an array of possible values of Q(x^i = H). No change by default.

        :x_vals: the costs of contribution corresponding to states xL and xH. No change by default.

        :num_quant: the number of quantized pts along an axis. No change by default.

        :T: time horizon. No change by default.

        :tol_dig: tolerant digit: set tolerance of belief to be 10^(-tol_dig). No change by default. Set tol_dig=-1 if not needed.

        Returns:
        ---
        None.

        """
        if type(q_vals) == type(None): q_vals = self.q_vals
        if type(x_vals) == type(None): x_vals = self.x_vals
        if type(num_quant) == type(None): num_quant = self.num_quant
        if type(T) == type(None): T = self.T
        if type(tol_dig) == type(None): tol_dig = self.tol_dig
        self.__init__(q_vals=q_vals, x_vals=x_vals, num_quant=num_quant, T=T, tol_dig=tol_dig)
        return

    def save(self, filename=None):
        """Save the current environment.

        Args:
        ---
        :filename: a file name. None by default. The environment will be saved as "filename".pkl in the current path.
        If filename is not given, it will use the current time.

        Returns:
        ---
        None.
        """
        if filename == None: filename = time.ctime()
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        return
    
    def load(self, filename):
        """Load a saved environment. This method will overwrite the current object.

        Args:
        ---
        :filename: a filename without ".pkl".

        Returns:
        ---
        None, but the current object will be replaced by the loaded object if success.
        """
        with open(filename + ".pkl", 'rb') as f:
            tmp_dict = pickle.load(f)
        
        self.__dict__.clear()
        self.__dict__.update(tmp_dict)

        return
    
    def reward(self, i, x0, x1, a0, a1):
        """Calculate the instantaneous reward.

        Args:
        ---
        :i: player's index.
        :x0: player 0's state.
        :x1: player 1's state.
        :a0: player 0's action.
        :a1: player 1's action.

        Returns:
        ---
        The instantaneous reward of player i given x0, x1, a0, a1.
        """
        return (a0 + a1 > 0) * 1 - (i == 0 and a0 == 1) * self.x_vals[x0] - \
                (i == 1 and a1 == 1) * self.x_vals[x1]
            

    def __find_belief_candidate(self):
        """Calculate all the possible pi^i_t values given the environment settings.
        """
        self.pi_vals[0] = np.array([q for q in self.q_vals])
        for t in range(1, self.T + 1):
            cand = [self.next_belief_val(pi, gammaL_ind, gammaH_ind, act) \
                    for pi in self.pi_vals[t - 1] \
                        for gammaL_ind in range(self.num_quant) \
                            for gammaH_ind in range(self.num_quant) \
                                for act in range(2)]
            self.pi_vals[t] = np.unique(cand)

        return
    
    def next_belief_val(self, pi, gammaL_ind, gammaH_ind, ai):
        """Calculate the value of the next belief.

        Args:
        ---
        :pi: the current belief on state H.

        :gammaL_ind: the index of the proposed prescription component gamma^i( | L).

        :gammaH_ind: the index of the proposed prescription component gamma^i( | H).

        :ai: action taken by current player i.

        Returns:
        ---
        :next_pi: the VALUE of the next belief.

        """
        gammaL_a = self.gamma_vals[gammaL_ind] * ai + (1 - self.gamma_vals[gammaL_ind]) * (1 - ai)
        gammaH_a = self.gamma_vals[gammaH_ind] * ai + (1 - self.gamma_vals[gammaH_ind]) * (1 - ai)
        den = pi * gammaH_a + (1 - pi) * gammaL_a # Prob of playing action act given pi, gammaL_ind, gammaH_ind
        if den == 0: # Deviation detected.
            if self.tol_dig < 0:
                return pi
            return round(pi, self.tol_dig)
        
        if self.tol_dig < 0:
            return pi * gammaH_a / den
        return round(pi * gammaH_a / den, self.tol_dig)
    
    def next_belief_ind(self, t, pi_ind, gammaL_ind, gammaH_ind, ai):
        """Calculate the index of the next belief.

        Args:
        ---
        :t: current time t.

        :pi_ind: the current belief on state H is self.pi_vals[t][pi_ind].

        :gammaL_ind: the index of the proposed prescription component gamma^i( | L).

        :gammaH_ind: the index of the proposed prescription component gamma^i( | H).

        :ai: action taken by the player.

        Returns:
        ---
        :next_pi_ind: the INDEX of the next belief.

        """
        next_pi = self.next_belief_val(self.pi_vals[t][pi_ind], gammaL_ind, gammaH_ind, ai)
        return np.where(self.pi_vals[t+1] == next_pi)[0][0]
    
    def backward(self, t, team=False):
        """Do the backward recursion for time t. Including the construction of the device,
        evaluate the value function, and find the expected social welfare to go.

        This function won't update the flags self.__complete and self.__team.

        Args:
        ---
        :t: current time t.

        :team: an indicator to solve it as a team problem. False by default.

        Returns:
        ---
        None, but the result matrices will be updated.
        """
        if t >= self.T or t < 0: return

        OB = self.obedient_matrices(t)
        V_mat = self.V_matrices(t, OB)
        SW_mat = self.SW_matrices(t, V_mat)

        if team: # Solve it as a team problem
            A_ub = None
            b_ub = None
        else: # Solve it as a game. 
            A_mat = -self.rational_matrices(t, OB) # The constraints in linprog take the form Ax <= 0 instead of >=.
        
        A_eq = np.ones((1, self.num_quant ** 4))
        b_eq = np.ones((1, 1))
        lb = np.zeros((self.num_quant ** 4, ))
        ub = np.ones((self.num_quant ** 4, ))

        for pi0_ind in range(len(self.pi_vals[t])):
            for pi1_ind in range(len(self.pi_vals[t])):
                if t == 0 and pi1_ind != pi0_ind: continue # if t == 0, we only consider the initial belief of the form [q, q].
                if not team:
                    A_ub = A_mat[:, :, :, :, :, pi0_ind, pi1_ind, :, :, :, :].squeeze().reshape(2 * self.num_quant * self.num_quant * 2 * 2, -1)
                    b_ub = np.zeros((A_ub.shape[0], ))
                c = -SW_mat[pi0_ind, pi1_ind, :, :, :, :].squeeze().reshape(-1, 1) # the optimizer solves the minimization problem.
                res = linprog(c, A_ub, b_ub, A_eq, b_eq, np.vstack([lb, ub]).T)
                if res.success:
                    self.phi[t][pi0_ind, pi1_ind, :, :, :, :] = res.x.reshape(self.num_quant, self.num_quant, self.num_quant, self.num_quant)
                else:
                    self.phi[t][pi0_ind, pi1_ind, :, :, :, :] = \
                        np.full(shape=(self.num_quant, self.num_quant, self.num_quant, self.num_quant), fill_value=np.nan)
                
        ind = np.indices((2, len(self.pi_vals[t]), len(self.pi_vals[t]), 2)) # (i, pi0_ind, pi1_ind, xi)
        V_ftn = lambda x: np.sum(V_mat[x[0], x[-1], x[1], x[2], :, :, :, :].squeeze() * self.phi[t][x[1], x[2], :, :, :, :].squeeze())
        self.V[t] = np.apply_along_axis(V_ftn, axis=0, arr=ind)

        if t > 0:
            self.social_welfare[t] = np.sum(SW_mat * self.phi[t], axis=(2, 3, 4, 5))
        else: # if t == 0, we only consider the initial belief of the form [q, q].
            self.social_welfare[t] = np.diagonal(np.sum(SW_mat * self.phi[t], axis=(2, 3, 4, 5)))

        return



    def __single_row_obedient(self, ind_list, t):
        """Compute single row of the coefficient matrices for the evaluation of obedient reward to go.

        Args:
        ---
        :ind_list: ind_list = [i, gammaiL_ind, gammaiH_ind, xi, pi0_ind, pi1_ind]

        :t: current time t.

        Returns:
        ---
        :coef_row: the coefficient vector (but indeed a matrix). 
            Format of coef_row: coef_row[gamma0L_ind, gamma0H_ind, gamma1L_ind, gamma1H_ind]
        """
        i, gammaiL_ind, gammaiH_ind, xi, pi0_ind, pi1_ind = ind_list.flatten()

        coef_row = np.zeros((self.num_quant, self.num_quant, self.num_quant, self.num_quant))

        
        for gamma_op_L_ind in range(self.num_quant):
            for gamma_op_H_ind in range(self.num_quant):
                if i == 0:
                    gamma0L_ind, gamma0H_ind = gammaiL_ind, gammaiH_ind
                    gamma1L_ind, gamma1H_ind = gamma_op_L_ind, gamma_op_H_ind
                else:
                    gamma0L_ind, gamma0H_ind = gamma_op_L_ind, gamma_op_H_ind
                    gamma1L_ind, gamma1H_ind = gammaiL_ind, gammaiH_ind

                gamma_ind = [[gamma0L_ind, gamma0H_ind], [gamma1L_ind, gamma1H_ind]]

                temp = 0
                for x_op in range(2):
                    pi_x_op = 0
                    if i == 0:
                        pi_x_op = self.pi_vals[t][pi1_ind] # Compute pi^{-i}(H)
                    else:
                        pi_x_op = self.pi_vals[t][pi0_ind]
                    if x_op == 0: pi_x_op = 1 - pi_x_op # Compute pi^{-i}(x_op)

                    for ai, a_op in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        if i == 0:
                            x0, x1 = xi, x_op
                            a0, a1 = ai, a_op
                        else:
                            x0, x1 = x_op, xi
                            a0, a1 = a_op, ai

                        gamma0_term = self.gamma_vals[gamma_ind[0][x0]]
                        if a0 == 0: gamma0_term = 1 - gamma0_term
                        gamma1_term = self.gamma_vals[gamma_ind[1][x1]]
                        if a1 == 0: gamma1_term = 1 - gamma1_term

                        next_pi0_ind = self.next_belief_ind(t, pi0_ind, gamma0L_ind, gamma0H_ind, a0)
                        next_pi1_ind = self.next_belief_ind(t, pi1_ind, gamma1L_ind, gamma1H_ind, a1)

                        temp += pi_x_op * gamma0_term * gamma1_term \
                            * (self.reward(i, x0, x1, a0, a1) \
                                + self.V[t+1][i, next_pi0_ind, next_pi1_ind, xi])
                        
                coef_row[gamma0L_ind, gamma0H_ind, gamma1L_ind, gamma1H_ind] += temp
        
        return coef_row

        


    def obedient_matrices(self, t):
        """Construct coefficient matrices for the evaluation of obedient reward to go at time t.

        Args:
        ---
        :t: current time t.

        Returns:
        ---
        :OB: Coefficient matrices for obedient reward to go at time t.
            Format of OB:
            OB[i, gammaiL_ind, gammaiH_ind, xi, \
                pi0_ind, pi1_ind, gamma0L_ind, gamma0H_ind, gamma1L_ind, gamma1H_ind]
        """
        ind = np.indices((2, self.num_quant, self.num_quant, 2, len(self.pi_vals[t]), len(self.pi_vals[t])))
        OB = np.apply_along_axis(lambda x: self.__single_row_obedient(x, t), axis=0, arr=ind)
        OB = OB.transpose((4, 5, 6, 7, 8, 9, 0, 1, 2, 3))
        
        return OB
    


    def __single_row_rational_to_substract(self, ind_list, t):
        """Compute single row of the coefficient matrices for rationality constraint.
        Please subtract it from the rationality matrix at hand.

        Args:
        ---
        :ind_list: ind_list = [i, gammaiL_ind, gammaiH_ind, xi, ai_dev, pi0_ind, pi1_ind]

        :t: current time t.

        Returns:
        ---
        :sub_row: the coefficient vector (but indeed a matrix). 
            Format of sub_row: sub_row[gamma0L_ind, gamma0H_ind, gamma1L_ind, gamma1H_ind]
        """

        i, gammaiL_ind, gammaiH_ind, xi, ai_dev, pi0_ind, pi1_ind = ind_list.flatten()

        # The following are copied from self.__single_row_obedient, with some necessary modifications.
        sub_row = np.zeros((self.num_quant, self.num_quant, self.num_quant, self.num_quant))

        
        for gamma_op_L_ind in range(self.num_quant):
            for gamma_op_H_ind in range(self.num_quant):
                if i == 0:
                    gamma0L_ind, gamma0H_ind = gammaiL_ind, gammaiH_ind
                    gamma1L_ind, gamma1H_ind = gamma_op_L_ind, gamma_op_H_ind
                else:
                    gamma0L_ind, gamma0H_ind = gamma_op_L_ind, gamma_op_H_ind
                    gamma1L_ind, gamma1H_ind = gammaiL_ind, gammaiH_ind

                gamma_ind = [[gamma0L_ind, gamma0H_ind], [gamma1L_ind, gamma1H_ind]]

                temp = 0
                for x_op in range(2):
                    pi_x_op = 0
                    if i == 0:
                        pi_x_op = self.pi_vals[t][pi1_ind] # Compute pi^{-i}(H)
                    else:
                        pi_x_op = self.pi_vals[t][pi0_ind]
                    if x_op == 0: pi_x_op = 1 - pi_x_op # Compute pi^{-i}(x_op)

                    for a_op in [0, 1]:
                        if i == 0:
                            x0, x1 = xi, x_op
                            a0, a1 = ai_dev, a_op # Fix a deviated action
                        else:
                            x0, x1 = x_op, xi
                            a0, a1 = a_op, ai_dev # Fix a deviated action

                        gamma0_term = self.gamma_vals[gamma_ind[0][x0]]
                        if a0 == 0: gamma0_term = 1 - gamma0_term
                        gamma1_term = self.gamma_vals[gamma_ind[1][x1]]
                        if a1 == 0: gamma1_term = 1 - gamma1_term

                        # Set gamma^i_t(ai_dev | x^i_t) to 1. The rest part does not change. 
                        if i == 0: gamma0_term = 1
                        else: gamma1_term = 1

                        next_pi0_ind = self.next_belief_ind(t, pi0_ind, gamma0L_ind, gamma0H_ind, a0)
                        next_pi1_ind = self.next_belief_ind(t, pi1_ind, gamma1L_ind, gamma1H_ind, a1)

                        temp += pi_x_op * gamma0_term * gamma1_term \
                            * (self.reward(i, x0, x1, a0, a1) \
                                + self.V[t+1][i, next_pi0_ind, next_pi1_ind, xi])
                        
                sub_row[gamma0L_ind, gamma0H_ind, gamma1L_ind, gamma1H_ind] += temp

        return sub_row

    def rational_matrices(self, t, OB=None):
        """Construct coefficient matrices for rationality constraints at time t.

        Args:
        ---
        :t: current time t.

        :OB: Coefficient matrices for obedient reward to go at time t. None by default.

        Returns:
        ---
        :A: Coefficient matrices for rationality constraints (no constraints on prob aspect.)
            Format of A: 
            A[i, gammaiL_ind, gammaiH_ind, xi, ai_dev, \
                pi0_ind, pi1_ind, gamma0L_ind, gamma0H_ind, gamma1L_ind, gamma1H_ind]
        """
        if type(OB) == type(None): OB = self.obedient_matrices(t)
        A = np.repeat(OB[:, :, :, :, np.newaxis, :, :, :, :, :, :], repeats=2, axis=4)
        ind = np.indices((2, self.num_quant, self.num_quant, 2, 2, len(self.pi_vals[t]), len(self.pi_vals[t])))
        A -= np.apply_along_axis(lambda x: self.__single_row_rational_to_substract(x, t),\
                                  axis=0, arr=ind).transpose((4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3))
        return A

    def V_matrices(self, t, OB=None):
        """Construct coefficient matrices for the update of value functions at time t.

        Args:
        ---
        :t: current time t.

        :OB: Coefficient matrices for obedient reward to go at time t. None by default.

        Returns:
        ---
        :C: Coefficient matrices for the evaluation of V.
            Format of C:
            C[i, xi, \
                pi0_ind, pi1_ind, gamma0L_ind, gamma0H_ind, gamma1L_ind, gamma1H_ind]
        """
        if type(OB) == type(None): OB = self.obedient_matrices(t)

        # Value function is the expected value of player's intermediate reward to go.
        # Accordingly, 
        # C[i, xi, pi0_ind, pi1_ind, gamma0L_ind, gamma0H_ind, gamma1L_ind, gamma1H_ind]
        # = \sum_{giL, giH} OB[i, giL, giH, xi, pi0_ind, pi1_ind, gamma0L_ind, gamma0H_ind, gamma1L_ind, gamma1H_ind]
        return np.sum(OB, axis=(1, 2))
    

    def SW_matrices(self, t, C=None):
        """Construct coefficient matrices for social welfare to go at time t.

        Args:
        ---
        :t: current time t.

        :C: Coefficient matrices for the update of value functions at time t. None by default.

        Returns:
        ---
        :S: Coefficient matrices for social welfare to go.
            Format of S:
            S[\
                pi0_ind, pi1_ind, gamma0L_ind, gamma0H_ind, gamma1L_ind, gamma1H_ind]
        """
        if type(C) == type(None): C = self.V_matrices(t)

        # We need a weighted sum of C. pi_vals[t] are the weights. 
        # To utilize the broadcast property, we need to reshape the pi_vals[t].
        pi0_vals_ad = self.pi_vals[t].reshape(-1, 1, 1, 1, 1, 1)
        pi1_vals_ad = self.pi_vals[t].reshape(1, -1, 1, 1, 1, 1)

        S = C[0, 0, :, :, :, :, :, :].squeeze() * (1 - pi0_vals_ad) + \
            C[0, 1, :, :, :, :, :, :].squeeze() * pi0_vals_ad + \
            C[1, 0, :, :, :, :, :, :].squeeze() * (1 - pi1_vals_ad) + \
            C[1, 1, :, :, :, :, :, :].squeeze() * pi1_vals_ad

        return S
    
    def solver(self, team=False, verbose=True, fig=True):
        """Solve the public investment problem using the backward recursion approach.

        Args:
        ---
        :team: solve it as a team problem if True. Otherwise, as a game. False by default.

        :verbose: print informations and status report if True. True by default.

        :fig: provide a plot of social welfare versus q if True (only applies to t=0 and verbose=True). True by default.

        Returns:
        ---
        None. A status report will be printed. 
        """
        if verbose: 
            print("The solver starts.")
            print()
            main_start = time.time()
        
        self.__complete = True
        for t in range(self.T - 1, -1, -1):
            start_t = time.time()
            self.backward(t, team)
            end_t = time.time()
            if verbose:
                print("Stage {} / {} complete... Time: {} seconds.".format(t, self.T - 1, end_t - start_t))
                print()
                self.status(t, fig)

        if not team:
            self.is_sPBE()

        if verbose:
            main_end = time.time()
            print("Total runtime: {} seconds".format(main_end - main_start))
            print()
        self.__team = team
        return
    
    def status(self, t=0, fig=True):
        """Print the status report of the solution if the solver has been run.

        Args:
        ---
        :t: time t. 0 by default.

        :fig: provide a plot of social welfare versus q if True (only applies to t=0). True by default.

        Returns:
        ---
        None. Print a status report.
        """
        if (not self.__complete) or t < 0 or t >= self.T: return
        print("The status of the solution at time {}:".format(t))
        print()
        if self.__team: print("Solver: Team Problem.")
        else: print("Solver: Game.")
        print()

        ind_failure = np.isnan(self.social_welfare[t].flatten())
        ind_success = np.logical_not(ind_failure)

        print("Success:")
        print("% of success: {}".format(ind_success.sum() / len(ind_success) * 100))

        print()

        print("Failures:")
        print("% of failure: {}".format(ind_failure.sum() / len(ind_failure) * 100))

        print()

        if t == 0:
            if ind_failure.any():
                print("Failure q's:")
                failure_qs = self.q_vals[ind_failure]
                print(failure_qs)
                print("Min: {}, Max: {}, Mean: {}, (Min+Max)/2: {}".format(failure_qs.min(), failure_qs.max(), failure_qs.mean(),\
                                                                            (failure_qs.min() + failure_qs.max()) / 2))
                print()
            else:
                print("No failure.")
                print()

            if ind_success.any():
                print("Success q's:")
                success_qs = self.q_vals[ind_success]
                print(success_qs)
                print("Expected social welfares:")
                print(self.social_welfare[t].flatten()[ind_success])
                print()
                if fig:
                    if self.__team: lab = "Social welfare: team"
                    else: lab = "Social welfare: sPCE"
                    plt.plot(self.q_vals, self.social_welfare[0].flatten(), label=lab)
                    plt.xlabel("Initial q values")
                    plt.ylabel("Social welfare")
                    plt.show()
            else:
                print("No success.")
                print()
            
    def compare(self):
        """Run another solver for the team problem if this instance is for game, 
        and run the solver for the game problem if this instance is for team.

        Args:
        ---
        None.

        Returns:
        ---
        None. A figure of social welfare will be provided, with both team and game data.
        """
        another = PublicInvestProblem(q_vals=self.q_vals, x_vals=self.x_vals, num_quant=self.num_quant, T=self.T, tol_dig=self.tol_dig)
        if not self.__complete: self.solver(fig=False)
        another.solver(team=not self.__team, verbose=True, fig=False)
        self.phi_comp = another.phi
        self.V_comp = another.V
        self.social_welfare_comp = another.social_welfare

        if self.__team: 
            social_team, social_game = self.social_welfare[0].flatten(), self.social_welfare_comp[0].flatten()
            self.sPBE_flag_comp = another.sPBE_flag
        else:
            social_game, social_team = self.social_welfare[0].flatten(), self.social_welfare_comp[0].flatten()
        plt.plot(self.q_vals, social_game, label = "Social welfare: sPCE")
        plt.plot(self.q_vals, social_team, label = "Social welfare: team")
        plt.xlabel("Initial q values")
        plt.ylabel("Social welfare")
        plt.legend()
        plt.show()

        self.__is_compared = True

        return
    
    def is_sPBE(self, t=0):
        """Check if a device at time t is sPBE or not for every given (pi1_ind, pi2_ind).

        Args:
        ---
        :t: time t of your interest.

        Returns:
        ---
        None.

        Update the self.sPBE_flag[t:] and self.__checked_sPBE[t:] as well.
        """
        if t >= self.T or t < 0: return
        if self.__checked_sPBE[t]: return 
        if t != self.T - 1 and self.__checked_sPBE[t + 1] == False: self.is_sPBE(t + 1)

        def is_next_sPBE(inputs):
            """(Local) Check if the phi_{t+1}[next_pi] is an sPBE or not.

            Args:
            ---
            :inputs: format: [pi0_ind, pi1_ind]

            Returns:
            ---
            True or False
            """
            pi0_ind, pi1_ind = inputs
            zero_bound = (self.tol_dig > 0) * (10 ** (-self.tol_dig))           
            # ind: g0L_ind, g0H_ind, g1L_ind, g1H_ind, a0, a1.
            ga_inds = np.indices((self.num_quant, self.num_quant, self.num_quant, self.num_quant, 2, 2)) 

            # For each (gamma, a) tuple, first check if gamma has nonzero prob. If not, we don't have to check sPBE for this tuple.
            is_0_measure = \
                lambda g_a_ind: \
                    (self.phi[t][pi0_ind, pi1_ind, g_a_ind[0], g_a_ind[1], g_a_ind[2], g_a_ind[3]] < zero_bound).flatten()
            # For (gamma, a) whose gamma can be recommended, check if the next phi adopts sPBE.
            if_next_is_sPBE = lambda g_a_ind: self.sPBE_flag[t+1][self.next_belief_ind(t, pi0_ind, g_a_ind[0], g_a_ind[1], g_a_ind[4]), \
                                                       self.next_belief_ind(t, pi1_ind, g_a_ind[2], g_a_ind[3], g_a_ind[5])].flatten()
            
            # There are two cases of sPBE: 1. gamma not taken at all, 2. gamma taken, but the next is sPBE.
            agg_ftn = lambda g_a_ind: np.logical_or(is_0_measure(g_a_ind), if_next_is_sPBE(g_a_ind))
            return np.apply_along_axis(agg_ftn, axis=0, arr=ga_inds).all()


        ind = np.indices((len(self.pi_vals[t]), len(self.pi_vals[t])))
        temp_ftn = lambda x: is_indep(self.phi[t][x[0], x[1], :, :, :, :].reshape(self.num_quant ** 2, self.num_quant ** 2))
        self.sPBE_flag[t] = np.apply_along_axis(temp_ftn, axis=0, arr=ind)

        if t < self.T - 1:
            self.sPBE_flag[t] = np.logical_and(self.sPBE_flag[t], \
                                                np.apply_along_axis(is_next_sPBE, axis=0, arr=ind))
            
        if t == 0:
            self.sPBE_flag[t] = np.diagonal(self.sPBE_flag[t])

        self.__checked_sPBE[t] = True

        return


    def detailed_plot(self, filename=None):
        """Plot a detailed figure of experiment results.
        It will do nothing if self.compare() hasn't been done.

        Args:
        ---
        :filename: Save the figure to a file named `filename`. None by default.

        Returns:
        ---
        None. Plot a figure.
        """        
        if not self.__is_compared: 
            print("Please run compare() first.")
            print()
            return
        
        if self.__team: 
            social_team, social_game = self.social_welfare[0].flatten(), self.social_welfare_comp[0].flatten()
            sPBE_markers = self.sPBE_flag_comp[0]
        else:
            social_game, social_team = self.social_welfare[0].flatten(), self.social_welfare_comp[0].flatten()
            sPBE_markers = self.sPBE_flag[0]
        plt.plot(self.q_vals, social_game, label = "Social welfare: sPCE")
        plt.plot(self.q_vals, social_team, label = "Social welfare: team")
        # plt.scatter(self.q_vals[sPBE_markers], social_game[sPBE_markers],\
        #              color="green", marker="x")
        # plt.scatter(self.q_vals[np.logical_not(sPBE_markers)], social_game[np.logical_not(sPBE_markers)],\
        #              color="red", marker="x")

        # Mark the sPBE and sPCE by fill_between.
        plt.fill_between(self.q_vals, social_game, np.zeros(social_game.shape), where=np.logical_not(np.isnan(social_game)), \
                         color='green', alpha=0.3, label="sPCE")
        # plt.fill_between(self.q_vals, social_game, np.zeros(social_game.shape), where=np.logical_not(sPBE_markers),\
        #                  color='red', alpha=0.3, label="sPCE")
        plt.fill_between(self.q_vals, social_game, np.zeros(social_game.shape), where=sPBE_markers, \
                         color='red', alpha=0.3, label="sPBE")
        

        plt.xlabel("Initial q values")
        plt.ylabel("Social welfare")
        plt.legend()
        if filename:
            plt.savefig(filename)
        plt.show()

        return



def is_indep(joint_dist, tolerant=1e-8):
    """Check if two random variables with the given joint distribution are independent or not.

    Args:
    ---
    :joint_dist: a 2 dimensional matrix. The joint distribution of two random variables.

    :tolerant: Tolerant. 1e-8 by default.

    Returns:
    ---
    :res: True if independent; False if not. 
    """
    if np.isnan(joint_dist).any(): return False

    x_margin = np.sum(joint_dist, axis=1).reshape(-1, 1)
    y_margin = np.sum(joint_dist, axis=0).reshape(1, -1)

    if (np.abs(joint_dist - x_margin * y_margin) < tolerant).all(): return True
    return False



if __name__ == "__main__":
    # env = PublicInvestProblem(q_vals=np.arange(0.01, 1, 0.01), num_quant=2, T=2)
    # env.compare()
    # env.save("0322")
    env = PublicInvestProblem(filename="0322")
    env.detailed_plot("0322.pdf")

    print()
