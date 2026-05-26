import numpy as np
import seaborn as sns
import pandas as pd
import random

class AllCustomerInformation(object):
    def __init__(self): # FILL IN INITIALIZATIONS

        # For sampling applicants from population
        self.linPop = None
        
        # Population applicant loan packages
        self.df_pop = None
        self.X_pop = None
        self.scores = None

        # Current sample of applicants in the system
        self.df_sample = None
        self.X_sample_adm = None
        self.X_sample_prop = None
        self.PaymentPool = None   # Information on existing loan (consists of 7 features: 1. customerID, 2. remaining balance on loan, 3. interest rate on loan, 4. time remaining to pay off loan, 5. Total payment requested, 6. Total amount paid, 7. Flag indicating whether account is in danger of default

        # People in loan pool
        self.sampleIDs = None
        self.df_pool = None
        self.X_pool_adm  = None
        self.X_pool_prop  = None
        self.X_pool = None

        # Current sample of customers waiting admissions decisions
        self.X_sample_adm_curr = None
        self.X_sample_prop_curr = None
        self.df_sample_curr = None

        # self.Potential_earnings_arr = None
        
        # Replace len(features_imported) with k_payment_pool

class AllBankInformation(object):
    def __init__(self): # FILL IN INITIALIZATIONS

        # Bank loan information
        self.bank_outstanding_loans = None     # Bank debts (deposits used to pay for loans + interest paid on those deposits. We abstract out the form of the deposits and just assume they all apply the same interest rate)
        self.bank_default_flag = None          # If bank defaulted
        self.bank_total_debt = None            # Total principal on all bank loans
        self.bank_next_payment = None          # Amount of banks next installment
        #self.bank_reject_all_flag = None

class TrackingIndicators(object):
    def __init__(self): # FILL IN INITIALIZATIONS

        # System indicators to monitor every round
        self.N_Applied = None                    # Total loans applicants system has received
        self.N_Applied_w = None
        self.N_Applied_nw = None

        self.N_Applied_static = None           # Number of applications in current time step
        self.N_Applied_w_static = None
        self.N_Applied_nw_static = None
        
        self.N_Accepted = None                 # Total applicant accepted into system
        self.N_Accepted_w = None
        self.N_Accepted_nw = None

        self.N_Accepted_static = None          # Number of applicants accepted into system at current time step
        self.N_Accepted_w_static = None  
        self.N_Accepted_nw_static = None 

        self.N_Disbursed = None                # Total loans disbursed to customers
        self.N_Disbursed_w = None                
        self.N_Disbursed_nw = None
        
        self.N_Disbursed_static = None         # Number of customers with funds disbursed at current time step
        self.N_Disbursed_w_static = None                
        self.N_Disbursed_nw_static = None
        
        self.N_Defaults = None                 # Total loans defaulted on by customers
        self.N_Defaults_w = None
        self.N_Defaults_nw = None

        self.N_Defaults_static = None          # Number loans terminated in default in current time step
        self.N_Defaults_w_static = None
        self.N_Defaults_nw_static = None
        
        self.N_Repaid = None                   # Total loans repaid by customers
        self.N_Repaid_w = None
        self.N_Repaid_nw = None

        self.N_Repaid_static = None            # Number of loans terminated in repayment in current time step
        self.N_Repaid_w_static = None
        self.N_Repaid_nw_static = None

        self.N_Repaid = None                   # Total loans repaid by customers
        self.N_Repaid_w = None
        self.N_Repaid_nw = None

        self.N_Score_Sum_static = None          # Sum of all applicants scores in current time step
        self.N_Score_Sum_w_static = None
        self.N_Score_Sum_nw_static = None

        self.N_Score_Sum = None                 # Total sum of all applicants scores over all time steps
        self.N_Score_Sum_w = None
        self.N_Score_Sum_nw = None

        self.WaitTimes_w = None
        self.WaitTimes_nw = None
        self.WaitTimes_w_static = None
        self.WaitTimes_nw_static = None
        
        self.Profits = None
        self.Profits_prev = None
        #self.MarginalProfits = None
        self.WaitTimes = None

        self.N_Terminated_w = None
        self.N_Terminated_nw = None

        #self.ds_t = None
        self.reward = None                                   # will append reward from each round and average over time steps
        self.full_reward = None
        self.Potential_earnings = None
        self.totals_4_installments = None

        # Total profits that could have been been potentially earned from all people who applied for loans
        self.Potential_earnings_arr = None
        self.Potential_earnings_curr_testing_01302024 = None
        
class EnvModelParameters(object):
    def __init__(self): # FILL IN INITIALIZATIONS

        # Propensity model parameters (The one chosen to be used. Either aware or unaware)
        self.W_prop = None
        self.b_prop = None

        # Propensity model parameters (unaware)
        self.W_prop_unsens = None
        self.b_prop_unsens = None

        # Propensity model parameters (aware)
        self.W_prop_sens = None
        self.b_prop_sens = None

        # Entrance decision model parameters (The one chosen to be used. Either aware or unaware)
        self.W_adm = None
        self.b_adm = None

        # Entrance decision model parameters (aware)
        self.W_adm_sens = None
        self.b_adm_sens = None

        # Entrance decision model parameters (unaware)
        self.W_adm_unsens = None
        self.b_adm_unsens = None

        # Queue related information
        self.QueueType = None
        self.Q = None                         # Queue for storing ordering of loan payouts

        # Constraints on model inputs and outputs
        self.eq_admit_flag = None
        self.eq_disburse_flag = None
        self.eq_debt_flag = None

        # Population distribution parameters used for normalizing features
        self.norm_params = None
        

# properties of agent entities
class AgentAdmit(object):
    def __init__(self, AwareOrNot):
        
        # Awareness of the sensitive attribute
        self.AwareOrNot = AwareOrNot
        
        # agent action structure (1. Produce threshold(s) 2. Produce score distribution(s))
        self.policyType = None
        
        # state
        self.state = None

        # augmented state
        self.aug_state = None
        
        # action
        self.action = None

        # Flags where agent is an RL model (True) or part of the environment (False)
        self.agent_flag = None
        
# properties of agent entities
class AgentDisburse(object):
    def __init__(self, AwareOrNot):
        
        # Awareness of the sensitive attribute
        self.AwareOrNot = AwareOrNot
        
        # agent action structure (Placeholder since always produces same action type 1. percentage from each group to admit each round)
        self.policyType = None
        
        # state
        self.state = None
        
        # augmented state
        self.aug_state = None
        
        # action
        self.action = None

        # Flags where agent is an RL model (True) or part of the environment (False)
        self.agent_flag = None

# properties of agent entities
class AgentDebt(object):
    def __init__(self, AwareOrNot):
        
        # Awareness of the sensitive attribute
        self.AwareOrNot = AwareOrNot
                
        # agent policy type (1. debt forgiveness (individual), 2. debt forgiveness (global), 3. interest rate adjustment, 4. maturity adjustment, ... include combos)
        self.policyType = None

        # state
        self.state = None

        # augmented state
        self.aug_state = None
        
        # action
        self.action = None

        # Flags where agent is an RL model (True) or part of the environment (False)
        self.agent_flag = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)

        # Object classes in the simulation
        self.agents = None                  # Each entry is an agent in the simulation 
        self.allcustomerinformation = None # ALL Customer related loan package information (for every module)
        self.allbankinformation = None
        self.trackingindicators = None
        self.envmodelparameters = None

        # Parameters of the simulation
        self.features_imported = None
        self.features_imported_sens = None
        self.features_imported_sens = None
        self.NPerPeriod = None
        self.perDefault = None
        self.perDefaultRecovery = None
        self.trainLoadAdmission = None
        self.trainLoadPropensity = None
        self.trainLoadPropensitySens = None
        self.save_dta_flag = None
        self.nn = None
        self.adm_freq = None
        self.aug_state_flag = None
        self.AwareOrNot = None

        # NOTE: agents may be subsumed into the environment (being replaced by traditional algorithms) depending on the paramter setups.
        #       This means the number of entries in self.agents my vary depending on the parameter setup of the simulation.


    # update state of the world
    def step(self, iter_num):

        # ------------------Apply Decisions-----------------
        # Apply debt forgiveness decisions
        self.apply_debt_decisions()
            
        # Apply disbursement decisions
        self.apply_disburse_decisions()
            
        # Apply admission decisions
        if (iter_num-1) % self.adm_freq == 0:
            self.apply_admission_decisions()
            # (This goes in the update to the agent's state)

        # Update environment (or do this in Env script?) (This should handle queueing stuff an swapping out agents to be subsumed into the environment)
        self.update_env()

        # ------------------Regular State Information-----------------
        # Update debt forgiveness agent state
        self.update_debt_agent_state()
        state2 = self.agents[2].state
            
        # Update disbursement decisions agent state
        self.update_disbursement_agent_state()
        state1 = self.agents[1].state
            
        # Update admission decisions agent state
        if iter_num % self.adm_freq == 0:
            self.update_admission_agent_state()
            state0 = self.agents[0].state
        else:
            _, N = self.agents[0].state.shape
            state0 = np.zeros((0,N))

        # ------------------Augmented State Information-----------------
        if self.aug_state_flag == True:
            self.aug_state(2)
            aug_state2 = self.agents[2].aug_state

            self.aug_state(1)
            aug_state1 = self.agents[1].aug_state

            self.aug_state(0)
            aug_state0 = self.agents[0].aug_state

            obs = [aug_state0, aug_state1, aug_state2]

        else:
            obs = [state0, state1, state2]
            
        # Things to return
        reward = self.trackingindicators.reward
        full_reward = self.trackingindicators.full_reward
        bank_default_flag = self.allbankinformation.bank_default_flag

        # End of episode
        if bank_default_flag or iter_num==798:
            dones = 1
        else:
            dones = 0

        return obs, reward, full_reward, bank_default_flag, dones

    
    #------------Augmenting each agent's state with additional information------------
    def aug_state(self, agent_i):

        # Aug
        aug = np.zeros((1,0)) # Augmented features
        
        for a_i in range(3):
            if a_i == agent_i:
                continue
            else:
                if len(self.agents[a_i].state)>0:
                    MM,NN = self.agents[a_i].state.shape
                    aug_ = np.mean(self.agents[a_i].state,0)
                    aug_ = np.append(aug_ ,MM/500) # Add number of loan packages as a feature
                    aug_ = np.expand_dims(aug_ ,0)
                    aug = np.hstack(( aug, aug_ ))
                else:
                    _,NN = self.agents[a_i].state.shape
                    aug = np.hstack((aug, np.zeros((1, NN + 1))))
        M,N = self.agents[agent_i].state.shape
        aug = np.repeat(aug,M,0)
        self.agents[agent_i].aug_state = np.hstack((self.agents[agent_i].state, aug))

    #------------------------------Env Update Function--------------------------------
    def update_env(self):

        # Import variables
        PaymentPool = self.allcustomerinformation.PaymentPool
        df_pop = self.allcustomerinformation.df_pop
        X_pop = self.allcustomerinformation.X_pop
        df_pool = self.allcustomerinformation.df_pool
        X_pool = self.allcustomerinformation.X_pool
        sampleIDs = self.allcustomerinformation.sampleIDs
        norm_params = self.envmodelparameters.norm_params
        Profits = self.trackingindicators.Profits
        Profits_prev = self.trackingindicators.Profits_prev
        reward = self.trackingindicators.reward
        full_reward = self.trackingindicators.full_reward
        
        WaitTimes = self.trackingindicators.WaitTimes
        WaitTimes_w = self.trackingindicators.WaitTimes_w
        WaitTimes_nw = self.trackingindicators.WaitTimes_nw
        WaitTimes_w_static = self.trackingindicators.WaitTimes_w_static
        WaitTimes_nw_static = self.trackingindicators.WaitTimes_nw_static
        
        N_Defaults = self.trackingindicators.N_Defaults
        N_Defaults_w = self.trackingindicators.N_Defaults_w
        N_Defaults_nw = self.trackingindicators.N_Defaults_nw
        N_Defaults_static = self.trackingindicators.N_Defaults_static
        N_Defaults_w_static = self.trackingindicators.N_Defaults_w_static
        N_Defaults_nw_static = self.trackingindicators.N_Defaults_nw_static
        
        N_Disbursed = self.trackingindicators.N_Disbursed
        N_Disbursed_w = self.trackingindicators.N_Disbursed_w
        N_Disbursed_nw = self.trackingindicators.N_Disbursed_nw
        N_Disbursed_static = self.trackingindicators.N_Disbursed_static
        N_Disbursed_w_static = self.trackingindicators.N_Disbursed_w_static
        N_Disbursed_nw_static = self.trackingindicators.N_Disbursed_nw_static
        
        N_Repaid = self.trackingindicators.N_Repaid
        N_Repaid_w = self.trackingindicators.N_Repaid_w
        N_Repaid_nw = self.trackingindicators.N_Repaid_nw
        N_Repaid_static = self.trackingindicators.N_Repaid_static
        N_Repaid_w_static = self.trackingindicators.N_Repaid_w_static
        N_Repaid_nw_static = self.trackingindicators.N_Repaid_nw_static
        
        N_Applied = self.trackingindicators.N_Applied
        N_Applied_w = self.trackingindicators.N_Applied_w
        N_Applied_nw = self.trackingindicators.N_Applied_nw
        N_Applied_static = self.trackingindicators.N_Applied_static
        N_Applied_w_static= self.trackingindicators.N_Applied_w_static
        N_Applied_nw_static = self.trackingindicators.N_Applied_nw_static
        
        N_Accepted = self.trackingindicators.N_Accepted
        N_Accepted_w = self.trackingindicators.N_Accepted_w
        N_Accepted_nw = self.trackingindicators.N_Accepted_nw
        N_Accepted_static = self.trackingindicators.N_Accepted_static
        N_Accepted_w_static = self.trackingindicators.N_Accepted_w_static
        N_Accepted_nw_static = self.trackingindicators.N_Accepted_nw_static

        N_Score_Sum = self.trackingindicators.N_Score_Sum
        N_Score_Sum_w = self.trackingindicators.N_Score_Sum_w
        N_Score_Sum_nw = self.trackingindicators.N_Score_Sum_nw
        N_Score_Sum_static = self.trackingindicators.N_Score_Sum_static
        N_Score_Sum_w_static = self.trackingindicators.N_Score_Sum_w_static
        N_Score_Sum_nw_static = self.trackingindicators.N_Score_Sum_nw_static

        N_Terminated_w = self.trackingindicators.N_Terminated_w
        N_Terminated_nw = self.trackingindicators.N_Terminated_nw
        
        totals_4_installments = self.trackingindicators.totals_4_installments
        Potential_earnings = self.trackingindicators.Potential_earnings
        Potential_earnings_arr = self.trackingindicators.Potential_earnings_arr
        Potential_earnings_curr_testing_01302024 = self.trackingindicators.Potential_earnings_curr_testing_01302024
        bank_outstanding_loans = self.allbankinformation.bank_outstanding_loans
        bank_default_flag = self.allbankinformation.bank_default_flag
        W_adm_sens = self.envmodelparameters.W_adm_sens
        b_adm_sens = self.envmodelparameters.b_adm_sens
        perDefault = self.perDefault

        # Come up with rule for default here and check if anyone defaulted. How much is lost in default? Everything
        # Check for default and repayment
        defaultIdx = ((PaymentPool[:,5]+1e-10)/(PaymentPool[:,4]+1e-10)< 1-perDefault) & (PaymentPool[:,6] == True)
        warningIdx = ((PaymentPool[:,5]+1e-10)/(PaymentPool[:,4]+1e-10)< 1-perDefault) & (PaymentPool[:,6] == False)
        repaidIdx  = PaymentPool[:,3]==0
        PaymentPool[warningIdx,6] = True            # Warn these customers about default
        defaultedLoans = PaymentPool[defaultIdx,:]  # Extract default customers
        repaidLoans = PaymentPool[repaidIdx,:]      # Extract repaid customers

        # Add remaining balance on repaid loans to bank profits (we assume that if they haven't dedaulted to this point, then they pay it all off)
        Profits += sum(PaymentPool[repaidIdx,1])
        MarginalProfits = Profits - Profits_prev
        Profits_prev = Profits
        

        # Update potential earnings so far
        Potential_earnings_curr = Potential_earnings_arr[0,0] + sum(Potential_earnings_arr[1:,0]*(Potential_earnings_arr[1:,2]/Potential_earnings_arr[1:,1]))

        #--------------Tally up defaults, repaid loans, and wait times--------------
        # Static
        N_Defaults_static = sum(defaultIdx)
        N_Defaults_w_static = sum((defaultIdx&(df_pool['race']==0)))
        N_Defaults_nw_static = sum((defaultIdx&(df_pool['race']==1)))
        N_Repaid_static = sum(repaidIdx)
        N_Repaid_w_static = sum((repaidIdx&(df_pool['race']==0)))
        N_Repaid_nw_static = sum((repaidIdx&(df_pool['race']==1)))
        N_Terminated_w_static = N_Repaid_w_static + N_Defaults_w_static
        N_Terminated_nw_static = N_Repaid_nw_static + N_Defaults_nw_static

        # Cumulative
        N_Defaults += N_Defaults_static
        N_Defaults_w += N_Defaults_w_static
        N_Defaults_nw += N_Defaults_nw_static
        N_Repaid += N_Repaid_static
        N_Repaid_w += N_Repaid_w_static
        N_Repaid_nw += N_Repaid_nw_static
        N_Terminated_w += N_Terminated_w_static
        N_Terminated_nw += N_Terminated_nw_static

        # Wait times
        dw_t = np.abs(WaitTimes_w/(N_Disbursed_w+1e-10) - WaitTimes_nw/(N_Disbursed_nw+1e-10))
        #--------------------------------------------------------------


        # Collect collateral from defaulted customers
        bank_outstanding_loans, remaining_collateral = self.recover_collateral(defaultedLoans, bank_outstanding_loans)  # The bank will take back collateral from defaulted customers, but it depreciates by 10% (bank gets 90% of outstanding balance.)
        Profits += remaining_collateral
        MarginalProfits += remaining_collateral

        # Update features in feature pool (the dataframe)
        df_pop, X_pop, df_pool, X_pool, PaymentPool, sampleIDs = self.feature_updates(df_pop, df_pool, X_pop, X_pool, defaultedLoans, repaidLoans, sampleIDs, defaultIdx, repaidIdx, PaymentPool, norm_params)
        
        # Default rate disparity
        df_t = np.abs((N_Defaults_w+1e-10)/(N_Terminated_w+1e-10) - (N_Defaults_nw+1e-10)/(N_Terminated_nw+1e-10)) # Default rate disparity

        # Score disparity (proxy for qualification rate)
        ds_t = np.abs((N_Score_Sum_w+1e-10)/(N_Applied_w+1e-10) - (N_Score_Sum_nw+1e-10)/(N_Applied_nw+1e-10))

        # Admissions rate disparity
        da_t = np.abs((N_Accepted_w+1e-10)/(N_Applied_w+1e-10) - (N_Accepted_nw+1e-10)/(N_Applied_nw+1e-10))
        
        # NOTE: In the case of unawareness, the MODELS should have the sensitive attribute removed. The METRICS
        # should not, though, since we are trying to capture reality. In other words, the true qualification rates
        # use ALL features to do the calculation, whether or not the model is not allowed to have access to the sensitive one.
        reward = np.array([[MarginalProfits, \
                                    N_Applied_w_static, N_Applied_nw_static, N_Accepted_w_static, N_Accepted_nw_static, \
                                    WaitTimes_w_static, WaitTimes_nw_static, N_Disbursed_w_static, N_Disbursed_nw_static, \
                                    N_Defaults_w_static, N_Defaults_nw_static, N_Score_Sum_w_static, N_Score_Sum_nw_static, \
                                    N_Terminated_w_static, N_Terminated_nw_static]])

        full_reward = np.array([[Profits, ds_t, df_t, dw_t, da_t]]) # These are cumulative, not isolated (we don't use them for model training but instead for indicator tracking.
        totals_4_installments = np.vstack((totals_4_installments, np.array([[0,0,0,0,0,0]])))
        Potential_earnings.append(Potential_earnings_curr)

        # If profits ever negative, immediately terminate (bank has gone under)
        if Profits < 0:
            bank_default_flag = True

        # Update variables
        self.allcustomerinformation.PaymentPool = PaymentPool
        self.allcustomerinformation.df_pop = df_pop
        self.allcustomerinformation.X_pop = X_pop
        self.allcustomerinformation.df_pool = df_pool
        self.allcustomerinformation.X_pool = X_pool
        self.allcustomerinformation.sampleIDs = sampleIDs
        self.trackingindicators.Profits = Profits
        self.trackingindicators.Profits_prev = Profits_prev
        self.trackingindicators.reward = reward
        self.trackingindicators.full_reward = full_reward
        self.trackingindicators.N_Defaults = N_Defaults
        self.trackingindicators.N_Defaults_w = N_Defaults_w
        self.trackingindicators.N_Defaults_nw = N_Defaults_nw
        self.trackingindicators.N_Repaid = N_Repaid
        self.trackingindicators.N_Repaid_w = N_Repaid_w
        self.trackingindicators.N_Repaid_nw = N_Repaid_nw
        self.trackingindicators.N_Terminated_w = N_Terminated_w
        self.trackingindicators.N_Terminated_nw = N_Terminated_nw
        self.trackingindicators.Potential_earnings = Potential_earnings
        self.trackingindicators.totals_4_installments = totals_4_installments
        self.allbankinformation.bank_outstanding_loans = bank_outstanding_loans
        self.allbankinformation.bank_default_flag = bank_default_flag
        

    #----------------------------State/Action Functions-------------------------------

    # Choose to admit new people into the loan system
    def apply_admission_decisions(self):

        # Import variables
        df_sample = self.allcustomerinformation.df_sample
        df_sample_curr = self.allcustomerinformation.df_sample_curr
        X_sample_adm = self.allcustomerinformation.X_sample_adm
        X_sample_adm_curr = self.allcustomerinformation.X_sample_adm_curr
        X_sample_prop = self.allcustomerinformation.X_sample_prop
        X_sample_prop_curr = self.allcustomerinformation.X_sample_prop_curr
        Q = self.envmodelparameters.Q
        admission_action_type = self.agents[0].policyType
        QueueType = self.envmodelparameters.QueueType
        eq_admit_flag = self.envmodelparameters.eq_admit_flag
        sampleIDs = self.allcustomerinformation.sampleIDs
        scores = self.allcustomerinformation.scores
        N_Applied = self.trackingindicators.N_Applied
        N_Applied_w = self.trackingindicators.N_Applied_w
        N_Applied_nw = self.trackingindicators.N_Applied_nw
        N_Accepted = self.trackingindicators.N_Accepted
        N_Accepted_w = self.trackingindicators.N_Accepted_w
        N_Accepted_nw = self.trackingindicators.N_Accepted_nw
        N_Score_Sum = self.trackingindicators.N_Score_Sum
        N_Score_Sum_w = self.trackingindicators.N_Score_Sum_w
        N_Score_Sum_nw = self.trackingindicators.N_Score_Sum_nw
        AwareOrNot = self.agents[0].AwareOrNot

        # Get action from RL agent
        if admission_action_type == 1:              # Policy outputs threshold on score distributions

            if AwareOrNot and not eq_admit_flag:    # Two thresholds (1 for each demographic group)

                if self.agents[0].agent_flag:
                    action = self.agents[0].action      # Get action from RL agent
                else:
                    action = np.array([[0.5, 0.5]])     # Always admit people with above average scores
                
                action_w = action[:, :1]
                action_nw = action[:, 1:2]
                race_w = df_sample_curr['race']==0
                race_nw = df_sample_curr['race']==1
                race_w = np.expand_dims(race_w,1)
                race_nw = np.expand_dims(race_nw,1)
                decisions = (((scores>action_w) & race_w) |  ((scores>action_nw) & race_nw)).flatten()
            else:                                   # One global threshold
                if self.agents[0].agent_flag:
                    action = self.agents[0].action  # Get action from RL agent
                else:
                    action = 0.5                    # Always admit people with above average scores
                    
                decisions = (scores>action).flatten()
                
        elif admission_action_type == 2:            # Policy directly outputs individual scores to rank admissions candidates
            scores = action
            decisions = (scores>0.5)
        else: # Action invalid
            raise ValueError

        # -------------Tallies stats-------------
        # Static
        N_Applied_static = len(scores)                                             # Tally total number of applicants in current time step
        N_Applied_w_static = len(scores[df_sample_curr['race']==0] )               # Tally total number of white applicants in current time step
        N_Applied_nw_static = len(scores[df_sample_curr['race']==1] )              # Tally total number of non-white applicants in current time step
        N_Accepted_static = sum(decisions)                                         # Tally total number of ACCEPTED applicants in current time step
        N_Accepted_w_static = sum((decisions&(df_sample_curr['race']==0)))         # Tally total number of white ACCEPTED applicants in current time step
        N_Accepted_nw_static = sum((decisions&(df_sample_curr['race']==1)))        # Tally total number of non-white ACCEPTED applicants in current time step
        N_Score_Sum_static = float(sum(scores))                                    # Sum of scores of all applicants applying in this round
        N_Score_Sum_w_static = float(sum(scores[df_sample_curr['race']==0] ))      # Sum of scores of all white applicants applying in this round
        N_Score_Sum_nw_static = float(sum(scores[df_sample_curr['race']==1] ))     # Sum of scores of all non-white applicants applying in this round

        # Cumulative
        N_Applied += N_Applied_static                                      # Tally total number of applicants so far
        N_Applied_w += N_Applied_w_static                                  # Tally total number of applicants so far
        N_Applied_nw += N_Applied_nw_static                                       # Tally total number of applicants so far
        N_Accepted += N_Accepted_static                                  # Tally total number of ACCEPTED applicants so far
        N_Accepted_w += N_Accepted_w_static                              # Tally total number of ACCEPTED applicants so far
        N_Accepted_nw += N_Accepted_nw_static                            # Tally total number of ACCEPTED applicants so far
        N_Score_Sum += N_Score_Sum_static                               # Sum of scores of all applicants applying in this round
        N_Score_Sum_w += N_Score_Sum_w_static                           # Sum of scores of all white applicants applying in this round
        N_Score_Sum_nw += N_Score_Sum_nw_static                         # Sum of scores of all non-white applicants applying in this round
        #----------------------------------------

        # Filter out rejected applicants
        X_sample_adm_curr  = X_sample_adm_curr[decisions]
        X_sample_prop_curr = X_sample_prop_curr[decisions]
        df_sample_curr = df_sample_curr[decisions]
        scores = scores[decisions]

        # Add accepted people into the system, filter out rejected people
        X_sample_adm = np.vstack((X_sample_adm,X_sample_adm_curr))
        X_sample_prop = np.vstack((X_sample_prop,X_sample_prop_curr))
        df_sample = pd.concat([df_sample, df_sample_curr], axis=0)     # df_sample = pd.concat([df_sample, df_sample_curr], ignore_index=True, axis=0) <--- bad we need the preserved id for when we remove Queue members
        races = df_sample_curr['race'] # races of newest people admitted
        sampleIDs_new = np.array(df_sample_curr.index)

        # Insert new samples into the Queue
        sampleIDs = np.concatenate((sampleIDs,sampleIDs_new))

        # Application package strength + fairness
        if QueueType == 'Fair':
            Q = self.QueueInsertion_Fair(Q, sampleIDs_new, scores, races) # Queue without fairness is based on a criterion like (1) highest score or (2) highest expected profit. (2) probably better, but for simplicity start with (1)

        # Application package strength
        else:
            Q = self.QueueInsertion(Q, sampleIDs_new, scores, races) # Queue without fairness is based on a criterion like (1) highest score or (2) highest expected profit. (2) probably better, but for simplicity start with (1)

        # Reorder sample data based on queue
        X_sample_adm = X_sample_adm[Q[4,:].astype(int)]
        X_sample_prop = X_sample_prop[Q[4,:].astype(int)]
        df_sample = df_sample.loc[Q[0,:].astype(int)]

        # Reset indices of Q members (indices...not order)
        Q[4,:] = np.linspace(0,len(Q[4,:])-1, len(Q[4,:])).astype(int)
                
        # Update variables
        self.allcustomerinformation.df_sample = df_sample
        self.allcustomerinformation.X_sample_adm = X_sample_adm
        self.allcustomerinformation.X_sample_prop = X_sample_prop
        self.envmodelparameters.Q = Q
        self.allcustomerinformation.sampleIDs = sampleIDs
        self.trackingindicators.N_Applied = N_Applied
        self.trackingindicators.N_Applied_w = N_Applied_w
        self.trackingindicators.N_Applied_nw = N_Applied_nw
        self.trackingindicators.N_Applied_static = N_Applied_static
        self.trackingindicators.N_Applied_w_static = N_Applied_w_static
        self.trackingindicators.N_Applied_nw_static = N_Applied_nw_static
        self.trackingindicators.N_Accepted = N_Accepted
        self.trackingindicators.N_Accepted_w = N_Accepted_w
        self.trackingindicators.N_Accepted_nw = N_Accepted_nw
        self.trackingindicators.N_Accepted_static = N_Accepted_static        
        self.trackingindicators.N_Accepted_w_static = N_Accepted_w_static
        self.trackingindicators.N_Accepted_nw_static = N_Accepted_nw_static
        self.trackingindicators.N_Score_Sum = N_Score_Sum
        self.trackingindicators.N_Score_Sum_w = N_Score_Sum_w
        self.trackingindicators.N_Score_Sum_nw = N_Score_Sum_nw
        self.trackingindicators.N_Score_Sum_static = N_Score_Sum_static        
        self.trackingindicators.N_Score_Sum_w_static = N_Score_Sum_w_static
        self.trackingindicators.N_Score_Sum_nw_static = N_Score_Sum_nw_static


    # Choose to admit new people into the loan system
    def update_admission_agent_state(self):

        # nn: number samples selected from the dataset

        X_pop = self.allcustomerinformation.X_pop
        linPop = self.allcustomerinformation.linPop
        df_pop = self.allcustomerinformation.df_pop
        W_adm = self.envmodelparameters.W_adm
        b_adm = self.envmodelparameters.b_adm
        sampleIDs = self.allcustomerinformation.sampleIDs
        X_sample_prop = self.allcustomerinformation.X_sample_prop
        X_sample_prop_curr = self.allcustomerinformation.X_sample_prop_curr
        reward = self.trackingindicators.reward
        Profits = self.trackingindicators.Profits
        Potential_earnings_arr = self.trackingindicators.Potential_earnings_arr
        nn = self.nn
        NPerPeriod = self.NPerPeriod
        AwareOrNot = self.AwareOrNot
        bank_total_debt = self.allbankinformation.bank_total_debt
        bank_next_payment = self.allbankinformation.bank_next_payment

        # Normalize bank features
        bank_total_debt = bank_total_debt/1e7
        bank_next_payment = bank_next_payment/1e5

        # Only sample people from population not currently in pool
        nn_curr = 0
        Pop_sample_idxs = np.array([])

        # If we can sample more than the number of applicants replaining in the population, sample them all.
        # Else randomly select those available to be sampled.
        if len(sampleIDs)+nn >= len(df_pop):
            Pop_sample_idxs = np.where(~df_pop.index.isin(sampleIDs)==True)[0]
        else:  
            while nn_curr!=nn:
                Pop_sample_idxs_curr = np.array(random.sample(linPop, nn))
                mask = np.invert(np.isin(df_pop.iloc[Pop_sample_idxs_curr].index,sampleIDs))
                Pop_sample_idxs = np.union1d(Pop_sample_idxs,Pop_sample_idxs_curr[mask]).astype('int')
                nn_curr = len(Pop_sample_idxs)
                if nn_curr>=nn:
                    Pop_sample_idxs = Pop_sample_idxs[:nn]
                    nn_curr = nn
            
        # Sampled data
        X_sample_adm_curr = X_pop['adm'][Pop_sample_idxs]
        X_sample_prop_curr = X_pop['prop'][Pop_sample_idxs]
        df_sample_curr = df_pop.iloc[Pop_sample_idxs]

        # Potential profits made on all sampled applicants
        Potential_earnings_arr = self.sample_potential_profits(Potential_earnings_arr, df_sample_curr, NPerPeriod)
            
        # Scores of sample applicants
        if AwareOrNot: # Aware
            scores = self.logistic(W_adm, b_adm, X_sample_adm_curr)
        else: # Unaware
            scores  = self.logistic(W_adm, b_adm, X_sample_adm_curr[:,:-1])
            
        # State variable calculation
        M, N = X_sample_adm_curr.shape
        augment = np.zeros((M,4))
        augment[:,0] = scores.flatten()
        reward_copy = reward.copy()
        reward_copy[0,0] = Profits/(1e7)# np.log(1+Profits)        
        augment[:,1:4] = np.repeat(reward_copy[:,:3],M,0)
        
        # Augment with bank features
        bank_augment = np.ones((M,2))
        bank_augment[:,0] = bank_augment[:,0]*bank_total_debt
        bank_augment[:,1] = bank_augment[:,0]*bank_next_payment
        augment = np.hstack( (augment, bank_augment) )

        X_sample_adm_aug = np.hstack(( X_sample_adm_curr, augment))
            
        # If unaware of sensitive attribute, remove it as a feature
        if not AwareOrNot: 
            X_sample_adm_aug = np.delete(X_sample_adm_aug, 9, 1)

        # Update admissions agent state
        self.agents[0].state = X_sample_adm_aug

        # Update Variables
        self.allcustomerinformation.X_sample_adm_curr = X_sample_adm_curr
        self.allcustomerinformation.X_sample_prop_curr = X_sample_prop_curr
        self.allcustomerinformation.df_sample_curr = df_sample_curr
        self.allcustomerinformation.scores = scores
        self.trackingindicators.Potential_earnings_arr = Potential_earnings_arr

    def update_disbursement_agent_state(self):

        # Import variables
        df_pop = self.allcustomerinformation.df_sample
        X_sample_adm = self.allcustomerinformation.X_sample_adm
        X_sample_prop = self.allcustomerinformation.X_sample_prop
        Q = self.envmodelparameters.Q
        reward = self.trackingindicators.reward
        AwareOrNot = self.AwareOrNot
        Profits = self.trackingindicators.Profits
        bank_total_debt = self.allbankinformation.bank_total_debt
        bank_next_payment = self.allbankinformation.bank_next_payment
        
        # Normalize bank features
        bank_total_debt = bank_total_debt/1e7
        bank_next_payment = bank_next_payment/1e5

        # Get indices of people waiting for funds
        #Q_idxsorted = Q[:, Q[0,:].astype(int) ]  # NEED THIS FOR augment[:,5] update to work below.
        #idxs = np.in1d(df_pop.index,Q_idxsorted[0,:])
        #X_waiting_adm = X_pop_adm[idxs]
        M, N = X_sample_adm.shape
        augment = np.zeros((M,5))               # 5th index is for monitoring waiting in the system
        augment[:,0] = Q[1,:].flatten()
        reward_copy = reward.copy()
        reward_copy[0,0] = Profits/(1e7)# np.log(1+Profits)
        augment[:,1:4] = np.repeat(reward_copy[:,:3],M,0)
        augment[:,4] = Q[2,:]/10         # NORMALIZE THIS

        # Augment with bank features
        bank_augment = np.ones((M,2))
        bank_augment[:,0] = bank_augment[:,0]*bank_total_debt
        bank_augment[:,1] = bank_augment[:,0]*bank_next_payment
        augment = np.hstack( (augment, bank_augment) )

        X_sample_adm_aug = np.hstack(( X_sample_adm, augment))

        # If unaware of sensitive attribute, remove it as a feature
        if not AwareOrNot: 
            X_sample_adm_aug = np.delete(X_sample_adm_aug, 9, 1)

        # Update disbursement agent state
        self.agents[1].state = X_sample_adm_aug

        ## Update variables  
        #self.envmodelparameters.Q = Q_idxsorted
                

    # Disburse loan funds to people waiting in the system (I.e. transfer from queue to repayment pool)
    def apply_disburse_decisions(self):

        # Import variables
        Q = self.envmodelparameters.Q
        NPerPeriod = self.NPerPeriod
        df_pool = self.allcustomerinformation.df_pool
        X_pool = self.allcustomerinformation.X_pool
        df_sample = self.allcustomerinformation.df_sample
        X_sample_adm = self.allcustomerinformation.X_sample_adm
        X_sample_prop = self.allcustomerinformation.X_sample_prop
        WaitTimes = self.trackingindicators.WaitTimes
        WaitTimes_w = self.trackingindicators.WaitTimes_w
        WaitTimes_nw = self.trackingindicators.WaitTimes_nw
        N_Disbursed = self.trackingindicators.N_Disbursed
        N_Disbursed_w = self.trackingindicators.N_Disbursed_w
        N_Disbursed_nw = self.trackingindicators.N_Disbursed_nw
        bank_outstanding_loans = self.allbankinformation.bank_outstanding_loans
        PaymentPool = self.allcustomerinformation.PaymentPool
        eq_disburse_flag = self.envmodelparameters.eq_disburse_flag
        AwareOrNot = self.agents[1].AwareOrNot
        disburse_action_type = self.agents[1].policyType

        # Intializations (Static)
        N_Disbursed_static = 0
        N_Disbursed_w_static = 0
        N_Disbursed_nw_static = 0
        WaitTimes['disbursed_w'] = []
        WaitTimes['disbursed_nw'] = []


        # If traditional algorithm is used, we use traditional queuing method applied in agent module
        # Otherwise, queue ordering based on agent's scores, which are accounted for here.
        if self.agents[1].agent_flag:
            if disburse_action_type == 1:
                score_ordering = self.agents[1].action.flatten()
                Q = Q[:,np.argsort(score_ordering)[::-1]]
                X_sample_adm = X_sample_adm[Q[4,:].astype(int)]
                X_sample_prop = X_sample_prop[Q[4,:].astype(int)]
                df_sample = df_sample.loc[Q[0,:].astype(int)]

                # Reset indices of Q members (indices...not order)
                Q[4,:] = np.linspace(0,len(Q[4,:])-1, len(Q[4,:])).astype(int)
            

                ## Ensure equal disbursement of funds across groups for given score ordering
                ## Can only force this as hard constaint if we know sensitive attribute.
                #if AwareOrNot and eq_disburse_flag:
                #    Q = self.Queue_eq_disburse(None, Q)
            elif disburse_action_type == 2 and AwareOrNot:
                p_w = self.agents[1].action # Percentage of white loan applicants whose funds are disbursed this round
                if len(p_w)>0:
                    p_w=p_w.item()
                
        # Disburse funds from queue
        _, Q_sz = Q.shape        
        _,k = PaymentPool.shape
        CurrPaymentPool = np.zeros((0,k))

        for n in range(NPerPeriod):
            
            # Only process queue if non-empty
            if Q_sz>0:

                if (disburse_action_type == 2 and AwareOrNot):
                    Q_nw = Q[:,Q[3,:]==1]
                    Q_w = Q[:,Q[3,:]==0]
                    M_w, N_w = Q_w.shape
                    M_nw, N_nw = Q_nw.shape
                    
                    if (N_Disbursed_w_static/(N_Disbursed_static+1e-10)> p_w or N_w == 0) and N_nw != 0:
                        customerID = int(Q_nw[0,0])
                        score = Q_nw[1,0]
                    else:
                        customerID = int(Q_w[0,0])
                        score = Q_w[1,0]
                else:
                    customerID = int(Q[0,0])
                    score = Q[1,0]
                    
                indexCurrCustomer = customerID==df_sample.index

                # Disburse funds to next customer. They are now in the payment pool and removed from the sample waiting to get their funds
                X_CurrCustomer_adm = X_sample_adm[indexCurrCustomer]
                X_CurrCustomer_prop = X_sample_prop[indexCurrCustomer]
                df_CurrCustomer = df_sample[indexCurrCustomer]
                X_sample_adm = X_sample_adm[~indexCurrCustomer]
                X_sample_prop = X_sample_prop[~indexCurrCustomer]
                df_sample = df_sample.loc[~indexCurrCustomer,:]


                # Get the features for this customer to add into their loan package
                RemainingBalance = float(df_CurrCustomer['loan_amnt']) # Funds of first person in the queue
                int_rate = float(df_CurrCustomer['int_rate'])
                months_remain = df_CurrCustomer['term'].values[0] # Time left to pay loan
                dti = float(df_CurrCustomer['dti'])
                mths_since_last_delinq = float(df_CurrCustomer['mths_since_last_delinq'])
                fico_range_low = float(df_CurrCustomer['fico_range_low'])
                fico_range_high = float(df_CurrCustomer['fico_range_high'])
                annual_inc = float(df_CurrCustomer['annual_inc'])
                installment = float(df_CurrCustomer['installment'])
                race = float(df_CurrCustomer['race'])

                # Update Wait times for disbursed loans
                if Q[3,0] == 0:
                    WaitTimes['disbursed_w'].append(Q[2,0]+1)
                else:
                    WaitTimes['disbursed_nw'].append(Q[2,0]+1)

                # Update queue
                N_Disbursed_static += 1 
                if race == 0:
                    N_Disbursed_w_static += 1
                else:
                    N_Disbursed_nw_static += 1 

#                if self.agents[1].agent_flag:
                if (disburse_action_type == 2 and AwareOrNot):
                    
                    idx = np.where(Q[0,:] == customerID)[0].item()
                    Q = np.delete(Q,idx,axis=1)
                    
                    if race == 0:
                        idx = np.where(Q_w[0,:] == customerID)[0].item()
                        Q_w = np.delete(Q_w, idx, axis=1)
                        
                    else:
                        idx = np.where(Q_nw[0,:] == customerID)[0].item()
                        Q_nw = np.delete(Q_nw, idx, axis=1)
                        
                else:
                    Q = Q[:,1:]
                    
                _, Q_sz = Q.shape
                
            else:
                break

            # Funds to disburse funds to customer
            X_pool["adm"] = np.vstack((X_pool["adm"], X_CurrCustomer_adm))
            X_pool["prop"] = np.vstack((X_pool["prop"], X_CurrCustomer_prop))
            df_pool = pd.concat([df_pool,df_CurrCustomer], axis=0)

            # Details on outstanding loan package                                                   x no need                    (      combine these two              )
            #                   Customer ID    ,   Remaining loan balance ,   interest rate  , months remaining on loan  ,  total requested payment,  total received payment,  warning close to default ,  *installment requested*,  dti, mths_since_last_delinq  , fico_range_low  , fico_range_high, annual income, race
            newLoan = np.array([[customerID   ,    RemainingBalance       ,   int_rate       , months_remain             ,        0                ,             0          ,        False              ,    installment          ,  dti, mths_since_last_delinq  , fico_range_low  , fico_range_high,  annual_inc, race, score]]) 
            CurrPaymentPool = np.vstack((CurrPaymentPool, newLoan))

        # Everyone who did not receive funds waits an extra round of processing
        if Q_sz>0:
            Q[2,:]+=1
            WaitTimes['non_disbursed_w'] = Q[2,Q[3,:]==0]
            WaitTimes['non_disbursed_nw'] = Q[2,Q[3,:]==1]

        # Bank "borrows" money
        bank_outstanding_loans = self.update_outstanding_bank_loans(bank_outstanding_loans, CurrPaymentPool)
        bank_total_debt, bank_next_payment = self.get_bank_debt_features(bank_outstanding_loans, f12=0.04)
        
        # Disburse funds to borrowers
        PaymentPool = np.vstack((PaymentPool, CurrPaymentPool))

        # Update total cumulative disbursements
        N_Disbursed += N_Disbursed_static
        N_Disbursed_w += N_Disbursed_w_static
        N_Disbursed_nw += N_Disbursed_nw_static

        # Update Wait Times
        WaitTimes_w_static = sum(WaitTimes['disbursed_w'])
        WaitTimes_nw_static = sum(WaitTimes['disbursed_nw'])
        WaitTimes_w  += WaitTimes_w_static
        WaitTimes_nw += WaitTimes_nw_static

        # Update variables
        self.envmodelparameters.Q = Q
        self.allcustomerinformation.df_pool = df_pool
        self.allcustomerinformation.X_pool = X_pool
        self.allcustomerinformation.df_sample = df_sample
        self.allcustomerinformation.X_sample_adm = X_sample_adm
        self.allcustomerinformation.X_sample_prop = X_sample_prop
        self.trackingindicators.WaitTimes = WaitTimes
        self.trackingindicators.WaitTimes_w_static = WaitTimes_w_static
        self.trackingindicators.WaitTimes_nw_static = WaitTimes_nw_static
        self.trackingindicators.WaitTimes_w = WaitTimes_w
        self.trackingindicators.WaitTimes_nw = WaitTimes_nw
        self.trackingindicators.N_Disbursed = N_Disbursed
        self.trackingindicators.N_Disbursed_w = N_Disbursed_w
        self.trackingindicators.N_Disbursed_nw = N_Disbursed_nw
        self.trackingindicators.N_Disbursed_static = N_Disbursed_static
        self.trackingindicators.N_Disbursed_w_static = N_Disbursed_w_static
        self.trackingindicators.N_Disbursed_nw_static = N_Disbursed_nw_static
        self.allbankinformation.bank_outstanding_loans = bank_outstanding_loans
        self.allbankinformation.bank_total_debt = bank_total_debt
        self.allbankinformation.bank_next_payment = bank_next_payment
        self.allcustomerinformation.PaymentPool = PaymentPool
        

    # Determine how much debt to forgive for people who have received loans
    def update_debt_agent_state(self):

        PaymentPool = self.allcustomerinformation.PaymentPool
        X_pool = self.allcustomerinformation.X_pool
        reward = self.trackingindicators.reward
        Profits = self.trackingindicators.Profits
        norm_params = self.envmodelparameters.norm_params
        AwareOrNot = self.agents[2].AwareOrNot
        bank_total_debt = self.allbankinformation.bank_total_debt
        bank_next_payment = self.allbankinformation.bank_next_payment
        
        # Normalize bank features            
        bank_total_debt = bank_total_debt/1e7
        bank_next_payment = bank_next_payment/1e5

        # Normalize features
        PaymentPool_aug = self.normalize_debt_features(PaymentPool, X_pool, reward[:,:3], Profits, bank_total_debt, bank_next_payment, norm_params)

        # If unaware of sensitive attribute, remove it as a feature
        if not AwareOrNot: 
            PaymentPool_aug = np.delete(PaymentPool_aug, 11, 1)

        # Update debt agent state
        self.agents[2].state = PaymentPool_aug


    # Determine how much debt to forgive for people who have received loans
    def apply_debt_decisions(self):

        # Import variables
        PaymentPool = self.allcustomerinformation.PaymentPool
        X_sample = self.allcustomerinformation.X_pool["prop"].astype('float')
        Profits = self.trackingindicators.Profits
        Potential_earnings_arr = self.trackingindicators.Potential_earnings_arr
        totals_4_installments = self.trackingindicators.totals_4_installments
        W_prop_sens = self.envmodelparameters.W_prop_sens
        b_prop_sens = self.envmodelparameters.b_prop_sens
        bank_outstanding_loans = self.allbankinformation.bank_outstanding_loans
        eq_debt_flag = self.envmodelparameters.eq_debt_flag
        debt_action_type = self.agents[2].policyType
        AwareOrNot = self.AwareOrNot

        # Current loan structures
        amt = PaymentPool[:,1] # Remaining balance on loans
        rr   = PaymentPool[:,2]/12 # Interest rates
        tau = PaymentPool[:,3] # Time left to pay loan

        # Apply debt forgiveness actions and update totals
        x_total = PaymentPool[:,7:8]

        if debt_action_type == 1: # Individual Debt Forgiveness

            if self.agents[2].agent_flag:
                action = self.agents[2].action
                x_forgiven = action*x_total
            else:
                x_forgiven = 0.0*x_total
            
        elif debt_action_type == 2: # Global Debt Forgiveness
            
            if AwareOrNot and not eq_debt_flag:       # different debt forgivness by sensitive attribute

                if self.agents[2].agent_flag:
                    action = self.agents[2].action
                    first_idx = np.array([True,False])
                    second_idx = np.array([False,True])
                    x_forgiven = np.zeros(x_total.shape)
                    x_forgiven[PaymentPool[:,-2]==0] = action[:, first_idx]*x_total[PaymentPool[:,-2]==0]
                    x_forgiven[PaymentPool[:,-2]==1] = action[:, second_idx]*x_total[PaymentPool[:,-2]==1]
                else:
                    x_forgiven = 0.0*x_total

            else: # same debt forgivness by sensitive attribute

                if self.agents[2].agent_flag:
                    action = self.agents[2].action
                    x_forgiven = action*x_total
                else:
                    x_forgiven = 0.0*x_total

        else: # Fix this to include other options like interest rate adjustment
            raise ValueError

            
        x_remain = x_total-x_forgiven
        x_remain[x_remain<0] = 0

        # Get customer propensities
        N,_ = X_sample.shape
        Noise = np.random.normal(0,0.025,(N,1)) # Noise added to payments since customers are not deterministic entities
        prop = X_sample@W_prop_sens+b_prop_sens + Noise
        prop[prop<0] = 0
        prop[prop>1] = 1

        # Customer payments
        Y = np.minimum(x_remain.flatten(), (prop*x_total).flatten())       # Amount paid by customer
        Y_total = x_forgiven.flatten() + Y  # Total installment amount (including debt forgiven on installment)

        # Bank update potential earnings
        Potential_earnings_arr[1:,2] += 1
        idxs_paid_off = Potential_earnings_arr[:,2] == Potential_earnings_arr[:,1]
        Earnings_paid_off_loans_curr = sum(Potential_earnings_arr[idxs_paid_off,0])
        Potential_earnings_curr_testing_01302024 = sum(Potential_earnings_arr[1:,0])
        Potential_earnings_arr[0,0] += Earnings_paid_off_loans_curr
        Potential_earnings_arr = np.delete(Potential_earnings_arr, idxs_paid_off, 0)

        # Update bank earnings
        Profits += sum(Y)
        bank_outstanding_loans, Profits = self.bank_pay_installments(bank_outstanding_loans, Profits)
        #MarginalProfits = sum(Y)
        #Profits += MarginalProfits
        #bank_outstanding_loans, Profits, MarginalProfits = self.bank_pay_installments(bank_outstanding_loans, Profits, MarginalProfits)

        # New balances (This is where the intervention should come in)
        amt=np.round((1+rr)*amt-Y_total,2)

        # Update Installment totals
        w_idx = PaymentPool[:,-2]==0
        nw_idx = PaymentPool[:,-2]==1
        totals_4_installments[-1, 0] += sum(PaymentPool[w_idx,7:8]) # Installments for w
        totals_4_installments[-1, 1] += sum(PaymentPool[nw_idx,7:8]) # Installments for nw
        totals_4_installments[-1, 2] += sum(x_forgiven[w_idx]) # Debt forgiven for w
        totals_4_installments[-1, 3] += sum(x_forgiven[nw_idx]) # Debt forgiven for nw
        totals_4_installments[-1, 4] += sum(Y[w_idx]) # Payments made for w
        totals_4_installments[-1, 5] += sum(Y[nw_idx]) # Payments made for nw

        # Update loan information
        PaymentPool[:,1] = amt
        PaymentPool[:,3] -=1
        PaymentPool[:,4] += x_total.flatten()
        PaymentPool[:,5] += Y_total.flatten()
        PaymentPool = self.update_installment(PaymentPool) # Installment for next month

        X_sample[:,0] = amt                 # Update loan balance for propenity predictions for next payment
        X_sample[:,8] -=1                   # Update months remaining for propenity predictions for next payment
        X_sample[:,7] = PaymentPool[:,7]    # Update monthly installment amount

        # Update variables
        self.allcustomerinformation.PaymentPool = PaymentPool
        self.allcustomerinformation.X_pool["prop"] = X_sample
        self.trackingindicators.Potential_earnings_arr = Potential_earnings_arr
        self.trackingindicators.Potential_earnings_curr_testing_01302024 = Potential_earnings_curr_testing_01302024
        self.trackingindicators.totals_4_installments = totals_4_installments
        self.trackingindicators.Profits = Profits
        #self.trackingindicators.MarginalProfits = MarginalProfits
        self.allbankinformation.bank_outstanding_loans = bank_outstanding_loans

    #----------------------------Auxiliary Functions-------------------------------
    def calcPopScoreDisparity(self, W, b, X, df):

        scores = self.logistic(W, b, X)
        scores_w = scores[df['race']==0]
        scores_nw = scores[df['race']==1]
        ds = np.abs(np.mean(scores_w) - np.mean(scores_nw))
    
        return ds

    # Sigmoid function
    def sig(self, x):
        return 1 / (1 + np.exp(-x))

    # Logistic Regression
    def logistic(self, W,b,X):
        return self.sig(X@W+b)

        
    def QueueInsertion(self, Q, IDs, scores, races):

        # Simple and greedy Queue update method
            
        # Reset indices of Q members (indices...not order)
        Q[4,:] = np.linspace(0,len(Q[4,:])-1, len(Q[4,:])).astype(int)
        
        IDs = IDs.flatten()
        IDs = np.expand_dims(IDs,0)
        scores = scores.flatten()
        scores = np.expand_dims(scores,0)
        WaitTime = np.zeros(scores.shape)

        # Fill max accuracy queue
        Q_addition = np.vstack((IDs,scores,WaitTime,races,idxs))

        # For indexing newly added additions to queue
        M,N = Q_addition.shape
        idxs = np.linspace(0,N-1,N)
        if Q_addition.shape[1] != 0:
            if Q.shape[1] == 0:
                MAX_idx = 0
            else:
                MAX_idx = np.max(Q[4,:])
        else:
            MAX_idx = np.array([])
        idxs = idxs + MAX_idx
        
        Q_addition = np.vstack((Q_addition, idxs.astype(int)))
        
        Q_addition = Q_addition[:,Q_addition[1,:].argsort()[::-1]]
        Q = np.hstack((Q,Q_addition))

        return Q

    def QueueInsertion_Fair(self, Q, IDs, scores, races):

        # Simple fair queuing update method

        # Reset indices of Q members (indices...not order)
        Q[4,:] = np.linspace(0,len(Q[4,:])-1, len(Q[4,:])).astype(int)
            
        IDs = IDs.flatten()
        IDs = np.expand_dims(IDs,0)
        scores = scores.flatten()
        scores = np.expand_dims(scores,0)
        WaitTime = np.zeros(scores.shape)

        # New members to be added to the queue
        Q_addition = np.vstack((IDs,scores,WaitTime,races))

        # For indexing newly added additions to queue
        M,N = Q_addition.shape
        idxs = np.linspace(0,N-1,N)
        if Q_addition.shape[1] != 0:
            if Q.shape[1] == 0:
                MAX_idx = 0
            else:
                MAX_idx = np.max(Q[4,:]) + 1
        else:
            MAX_idx = np.array([])
        idxs = idxs + MAX_idx

        Q_addition = np.vstack((Q_addition, idxs.astype(int)))

        # Sort new members added to queue
        Q_addition = Q_addition[:,Q_addition[1,:].argsort()[::-1]]

        # Make sure queue disburses at equal rates across groups
        Q = self.Queue_eq_disburse(Q, Q_addition)
        
        return Q

    def Queue_eq_disburse(self, Q, Q_addition):

        ''' Assuming Q is already ordered to satisfy equal rate of
            disbursement of funds across groups, and Q_addition is not,
            we sort Q_addition so that to satisfy equal rate of
            disbursement of funds and then append to Q. If Q is empty
            we return Q_addition.'''


        # Get queues for each group
        Q_w = Q_addition[:,Q_addition[3,:]==0]
        Q_nw = Q_addition[:,Q_addition[3,:]==1]
        _, N_w = Q_w.shape
        _, N_nw = Q_nw.shape
        t_w = np.ones(N_w)/N_w
        t_nw = np.ones(N_nw)/N_nw

        # CDF calculation for each group
        c_w  = np.cumsum(t_w)
        c_nw = np.cumsum(t_nw)

        # Create the fair queue
        N = N_w + N_nw
        Q_fair = np.zeros((5,N))

        for ii in range(N):

            # Case: both queues are non-empty
            if (len(c_w) > 0) & (len(c_nw) > 0):

                # Choose from nw queue
                if c_nw[0]<=c_w[0]:
                    Q_fair[:,ii] = Q_nw[:,0]
                    Q_nw = Q_nw[:,1:]
                    c_nw = c_nw[1:]

                # Choose from w queue
                else:
                    Q_fair[:,ii] = Q_w[:,0]
                    Q_w = Q_w[:,1:]
                    c_w = c_w[1:]

            # If w queue empty, only nw members left. Fill remainder of queue with them.      
            elif len(c_w) == 0:
                Q_fair[:,ii:] = Q_nw
                break

            # If nw queue empty, only w members left. Fill remainder of queue with them.   
            else:
                Q_fair[:,ii:] = Q_w
                break
            
        # Update queue
        if Q is not None: 
            Q = np.hstack((Q,Q_fair))
        else:
            Q = Q_fair

        return Q


    def get_bank_debt_features(self, bank_outstanding_loans, f12=0.04):

        # Total principal on bank loans
        Balance = np.sum(bank_outstanding_loans[:,0])

        # Amount of next bank installment (log scale)
        N, _= bank_outstanding_loans.shape # number of loans outstanding by bank
        installment = 0
        for ii in range(N):

            # Calculate installment for current loans ##FIX THIS FOR THE LAST PAYMENT THAT NEEDS TO BE REMOVED
            balance_i = bank_outstanding_loans[ii,0]
            tau_i = bank_outstanding_loans[ii,1]
            f_i = bank_outstanding_loans[ii,2]
            alp_N = (1+f_i)**tau_i
            installment += max( [min([(alp_N*f_i*balance_i)/(alp_N-1), balance_i]), 0] )

        return Balance, installment
        

    def update_outstanding_bank_loans(self, bank_outstanding_loans, newLoans, f12= 0.04):

        # If no new loans disbursed, then bank doesn't need to borrow
        if newLoans.shape[0] == 0:
            return bank_outstanding_loans

        #bank_outstanding_loans: Nxk, where N is number of outstanding loans and k is the terms/status of the loan [installment, remaining_time_to_pay]
        Balances = newLoans[:,1]
        int_rates = newLoans[:,2]
        term = newLoans[:,3]
        f = f12/12    # annual interest rate amortized to monthly interest rate

        # Distinguish length of loans
        Balance36 = sum(Balances[term==36])
        Balance60 = sum(Balances[term==60])

        # Update loans
        new_loans = np.array([[Balance36, 36, f], [Balance60, 60, f]])              # Track balances and number of months left on each bank loan
        bank_outstanding_loans = np.vstack((bank_outstanding_loans,new_loans))

        return bank_outstanding_loans

    def sample_potential_profits(self, Potential_earnings_arr, df_sample_curr, NPerPeriod, f12= 0.04):

        # This produces the total number of samples that can be made on a given loan application for a given number of time steps.
        nn, _ = df_sample_curr.shape

        #---------Calculate potential earnings on all loans--------

        Balances =  np.expand_dims(df_sample_curr['loan_amnt'].to_numpy(), 1)
        # Total amount customer must pay
        rr = np.expand_dims(df_sample_curr['int_rate'].to_numpy(), 1)
        tau = np.expand_dims(df_sample_curr['term'].to_numpy(), 1)
        rr = rr/12
        alp = 1+rr
        total_customer_payment = tau*((alp**tau)/(alp**tau-1))*rr*Balances

        # Total amount bank must pay
        rr_bank = f12/12
        alp_bank = 1+rr_bank
        total_bank_payment = tau*((alp_bank**tau)/(alp_bank**tau-1))*rr_bank*Balances
    
        Potential_earnings_arr_curr = total_customer_payment - total_bank_payment
        Potential_earnings_arr_curr = np.hstack((Potential_earnings_arr_curr,tau, np.zeros((nn,1))))

        return np.vstack((Potential_earnings_arr,Potential_earnings_arr_curr))

    def update_installment(self, PaymentPool):

        # Current loan structures
        amt = PaymentPool[:,1] # Remaining balance on loans
        rr   = PaymentPool[:,2]/12 # Interest rates amortized monthly
        tau = PaymentPool[:,3] # Time left to pay loan

        PaymentPool[:,7] = (amt*rr*(1+rr)**tau)/((1+rr)**tau-1)
        PaymentPool[:,7][PaymentPool[:,7]==np.inf] = 0
        PaymentPool[:,7][PaymentPool[:,7]==-np.inf] = 0
        PaymentPool[:,7][np.isnan(PaymentPool[:,7])] = 0
    
        return PaymentPool

    def bank_pay_installments(self, bank_outstanding_loans, Profits):

        # Bank reduces profits by installments paid to its depositors
        N, _= bank_outstanding_loans.shape # number of loans outstanding by bank

        delete_idxs = []
        for ii in range(N):

            # Calculate installment for current loans ##FIX THIS FOR THE LAST PAYMENT THAT NEEDS TO BE REMOVED
            Balance = bank_outstanding_loans[ii,0]
            tau = bank_outstanding_loans[ii,1]
            f = bank_outstanding_loans[ii,2]
            alp_N = (1+f)**tau
            installment = max( [min([(alp_N*f*Balance)/(alp_N-1), Balance]), 0] )
            Profits -= installment

            # Update balance and time left on loan
            bank_outstanding_loans[ii,0] = Balance-installment
            bank_outstanding_loans[ii,1] -= 1

            # If loan paid off, remove it from bank's outstanding loans
            if (bank_outstanding_loans[ii,0]==0) | (bank_outstanding_loans[ii,1]==0):
                delete_idxs.append(ii)
            
        # Delete paid off loans
        bank_outstanding_loans = np.delete(bank_outstanding_loans,delete_idxs,axis=0)
        
        return bank_outstanding_loans, Profits


    def recover_collateral(self, defaultedLoans, bank_outstanding_loans):

        N_loans,_ = bank_outstanding_loans.shape
        collateral_recovery = np.sum( defaultedLoans[:,1] )*0.9
        for ii in range(N_loans):
            if collateral_recovery>0:
                payback_amount = np.min([bank_outstanding_loans[ii,0], collateral_recovery])
                bank_outstanding_loans[ii,0] -= payback_amount
                collateral_recovery -= payback_amount
        
        return bank_outstanding_loans, collateral_recovery


    def normalize_debt_features(self, PaymentPool, X_pool, reward, Profits, bank_total_debt, bank_next_payment, norm_params):

        M, N = PaymentPool.shape
        
        # Augment with rewards
        augment = reward.copy()
        augment[0,0] = Profits/(1e7)# np.log(1+Profits)
        augment = np.repeat(augment,M,0)

        # Augment with bank features
        bank_augment = np.ones((M,2))
        bank_augment[:,0] = bank_augment[:,0]*bank_total_debt
        bank_augment[:,1] = bank_augment[:,0]*bank_next_payment
        augment = np.hstack( (augment, bank_augment) )
                  
        #augment[:,0] = np.min([augment[:,0]/(PaymentPool[:,7]+0.0000001, np.ones(PaymentPool[:,7].shape))]) # Normalizing profit-installment relation to communicate to model
        PaymentPool_aug = np.hstack((PaymentPool[:,1:], augment)) # We remove the customer IDs from PaymentPool
        PaymentPool_aug = np.hstack((PaymentPool_aug,X_pool['adm'][:,:1]))  # Add normalized amount of loan
        PaymentPool_aug[:,0] = PaymentPool_aug[:,0]/X_pool['prop'][:,0]    # Percentage left to pay on loan
        PaymentPool_aug[:,3] = (PaymentPool_aug[:,4]+0.00000001)/(PaymentPool_aug[:,3]+0.00000001) # Amount recieved/amount requested
        PaymentPool_aug[:,1] = X_pool['adm'][:,6]  # Normalize interest rate 
        PaymentPool_aug[:,2] = PaymentPool_aug[:,2]/X_pool['prop'][:,8]  # (remaining time till maturity)
        PaymentPool_aug[:,7] = X_pool['adm'][:,2]  # dti
        PaymentPool_aug[:,8] = X_pool['adm'][:,3]  # mths_since_last_delinq
        PaymentPool_aug[:,9] = X_pool['adm'][:,4]  # fico_range_low
        PaymentPool_aug[:,10] = X_pool['adm'][:,5]  # fico_range_high
        PaymentPool_aug[:,11] = X_pool['adm'][:,1]  # annual_inc
        PaymentPool_aug[:,6] = (PaymentPool_aug[:,6] - norm_params['mu_test'][7])/norm_params['sig_test'][7]  # installment
        PaymentPool_aug = np.delete(PaymentPool_aug, 4, 1) # Delete
        
        return PaymentPool_aug


    def feature_updates(self, df_pop, df_pool, X_pop, X_pool, defaultedLoans, repaidLoans, sampleIDs, defaultIdx, repaidIdx, PaymentPool, norm_params):

        # Heuristic method for updating features given you did or did not repay your loans

        # Dataframe population updates based on terminated loans
        # Fico score for defaulted loans drop by 50. Delinquincies in past 2 years increases.
        df_pop.loc[np.in1d(df_pop.index,defaultedLoans[:,0]), "fico_range_low"]-=100
        df_pop.loc[np.in1d(df_pop.index,defaultedLoans[:,0]), "fico_range_high"]-=100
        df_pop.loc[np.in1d(df_pop.index,defaultedLoans[:,0]), "mths_since_last_delinq"]-=5

        X_pop["adm"][np.in1d(df_pop.index,defaultedLoans[:,0]), 3] = (np.minimum(np.maximum(X_pop["adm"][np.in1d(df_pop.index,defaultedLoans[:,0]), 3]*norm_params["sig_test"][3]-5+norm_params["mu_test"][3],0),25)-norm_params["mu_test"][3])/norm_params["sig_test"][3]
        X_pop["adm"][np.in1d(df_pop.index,defaultedLoans[:,0]), 4] = (np.minimum(np.maximum(X_pop["adm"][np.in1d(df_pop.index,defaultedLoans[:,0]), 4]*norm_params["sig_test"][4]-100+norm_params["mu_test"][4],300),850)-norm_params["mu_test"][4])/norm_params["sig_test"][4]
        X_pop["adm"][np.in1d(df_pop.index,defaultedLoans[:,0]), 5] = (np.minimum(np.maximum(X_pop["adm"][np.in1d(df_pop.index,defaultedLoans[:,0]), 5]*norm_params["sig_test"][5]-100+norm_params["mu_test"][5],300),850)-norm_params["mu_test"][5])/norm_params["sig_test"][5]

        X_pop["prop"][np.in1d(df_pop.index,defaultedLoans[:,0]), 3] -= 5
        X_pop["prop"][np.in1d(df_pop.index,defaultedLoans[:,0]), 4] -= 100
        X_pop["prop"][np.in1d(df_pop.index,defaultedLoans[:,0]), 5] -= 100
    
        # Fico score for repaid loans improve by 50. Delinquincies in past 2 years decrease.
        df_pop.loc[np.in1d(df_pop.index,repaidLoans[:,0]), "fico_range_low"]+=100
        df_pop.loc[np.in1d(df_pop.index,repaidLoans[:,0]), "fico_range_high"]+=100
        df_pop.loc[np.in1d(df_pop.index,repaidLoans[:,0]), "mths_since_last_delinq"]+=5

        X_pop["adm"][np.in1d(df_pop.index,repaidLoans[:,0]), 3] = (np.minimum(np.maximum(X_pop["adm"][np.in1d(df_pop.index,repaidLoans[:,0]), 3]*norm_params["sig_test"][3]+5+norm_params["mu_test"][3],0),25)-norm_params["mu_test"][3])/norm_params["sig_test"][3]
        X_pop["adm"][np.in1d(df_pop.index,repaidLoans[:,0]), 4] = (np.minimum(np.maximum(X_pop["adm"][np.in1d(df_pop.index,repaidLoans[:,0]), 4]*norm_params["sig_test"][4]+100+norm_params["mu_test"][4],300),850)-norm_params["mu_test"][4])/norm_params["sig_test"][4]
        X_pop["adm"][np.in1d(df_pop.index,repaidLoans[:,0]), 5] = (np.minimum(np.maximum(X_pop["adm"][np.in1d(df_pop.index,repaidLoans[:,0]), 5]*norm_params["sig_test"][5]+100+norm_params["mu_test"][5],300),850)-norm_params["mu_test"][5])/norm_params["sig_test"][5]

        X_pop["prop"][np.in1d(df_pop.index,repaidLoans[:,0]), 3] += 5
        X_pop["prop"][np.in1d(df_pop.index,repaidLoans[:,0]), 4] += 100
        X_pop["prop"][np.in1d(df_pop.index,repaidLoans[:,0]), 5] += 100

        # Clip variables
        df_pop.loc[df_pop["mths_since_last_delinq"]<0, "mths_since_last_delinq"]=0
        df_pop.loc[df_pop["mths_since_last_delinq"]>25, "mths_since_last_delinq"]=25
        df_pop.loc[df_pop["fico_range_low"]<300, "fico_range_low"] = 300
        df_pop.loc[df_pop["fico_range_high"]<300, "fico_range_high"] = 300
        df_pop.loc[df_pop["fico_range_low"]>850, "fico_range_low"] = 850
        df_pop.loc[df_pop["fico_range_high"]>850, "fico_range_high"] = 850

        X_pop["prop"][X_pop["prop"][:,3]<0, 3] = 0
        X_pop["prop"][X_pop["prop"][:,3]>25, 3] = 25
        X_pop["prop"][X_pop["prop"][:,4]<300, 4] = 300
        X_pop["prop"][X_pop["prop"][:,4]>850, 4] = 850
        X_pop["prop"][X_pop["prop"][:,5]<300, 5] = 300
        X_pop["prop"][X_pop["prop"][:,5]>850, 5] = 850
    
    
        # Remove terminated loans from payment pool
        removeIdx = defaultIdx | repaidIdx
        PaymentPool = np.delete(PaymentPool,removeIdx,axis=0)
        df_pool = df_pool[np.invert(removeIdx)]
        X_pool['adm'] = X_pool['adm'][np.invert(removeIdx)]
        X_pool['prop'] = X_pool['prop'][np.invert(removeIdx)]

        # Remove terminated loan clients from list of Ids of clients repaying loans
        sampleIDs = np.setdiff1d(sampleIDs,defaultedLoans[:,0])
        sampleIDs = np.setdiff1d(sampleIDs,repaidLoans[:,0])

        return df_pop, X_pop, df_pool, X_pool, PaymentPool, sampleIDs
