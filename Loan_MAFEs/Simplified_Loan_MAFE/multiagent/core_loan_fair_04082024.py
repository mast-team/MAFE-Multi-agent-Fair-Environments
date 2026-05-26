import numpy as np
import seaborn as sns
import pandas as pd
import random
from numpy.random import  binomial, uniform

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

        # Current sample of customers waiting admissions decisions
        self.X_sample_adm_curr = None
        self.X_sample_prop_curr = None
        self.df_sample_curr = None

        
class AllBankInformation(object):
    def __init__(self): # FILL IN INITIALIZATIONS

        # Bank loan information
        self.bank_default_flag = None          # If bank defaulted

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
        self.WaitTimes = None

        self.N_Terminated_w = None
        self.N_Terminated_nw = None

        #self.ds_t = None
        self.reward = None                                   # will append reward from each round and average over time steps
        self.full_reward = None
        
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
        self.AwareOrNot = None

        # NOTE: agents may be subsumed into the environment (being replaced by traditional algorithms) depending on the paramter setups.
        #       This means the number of entries in self.agents my vary depending on the parameter setup of the simulation.


    # update state of the world
    def step(self, iter_num):

        # Update environment (or do this in Env script?) (This should handle queueing stuff an swapping out agents to be subsumed into the environment)
        self.apply_decisions()

        # ------------------Regular State Information-----------------
        # Update agent states
        self.update_agent_states()
        state0 = self.agents[0].state
        state1 = self.agents[1].state
        state2 = self.agents[2].state

        # ------------------Augmented State Information-----------------
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
    
    #------------------------------Env Update --------------------------------
    # Choose to admit new people into the loan system
    def apply_decisions(self):

        # Import variables (adm)
        df_sample = self.allcustomerinformation.df_sample
        X_sample_adm = self.allcustomerinformation.X_sample_adm
        X_sample_prop = self.allcustomerinformation.X_sample_prop
        eq_admit_flag = self.envmodelparameters.eq_admit_flag
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

        # Import variables (disburse)
        WaitTimes = self.trackingindicators.WaitTimes
        WaitTimes_w = self.trackingindicators.WaitTimes_w
        WaitTimes_nw = self.trackingindicators.WaitTimes_nw
        N_Disbursed = self.trackingindicators.N_Disbursed
        N_Disbursed_w = self.trackingindicators.N_Disbursed_w
        N_Disbursed_nw = self.trackingindicators.N_Disbursed_nw


        # Import variables (debt)
        N_Defaults = self.trackingindicators.N_Defaults
        N_Defaults_w = self.trackingindicators.N_Defaults_w
        N_Defaults_nw = self.trackingindicators.N_Defaults_nw
        N_Repaid = self.trackingindicators.N_Repaid
        N_Repaid_w = self.trackingindicators.N_Repaid_w
        N_Repaid_nw = self.trackingindicators.N_Repaid_nw
        N_Terminated_w = self.trackingindicators.N_Terminated_w
        N_Terminated_nw = self.trackingindicators.N_Terminated_nw
        W_prop_sens = self.envmodelparameters.W_prop_sens
        b_prop_sens = self.envmodelparameters.b_prop_sens
        Profits_prev = self.trackingindicators.Profits
        eq_debt_flag = self.envmodelparameters.eq_debt_flag

        # Environment variables
        norm_params = self.envmodelparameters.norm_params
        df_pop = self.allcustomerinformation.df_pop
        X_pop = self.allcustomerinformation.X_pop


        #---------ADM actions-----------

        # Get action from RL agent
        if AwareOrNot and not eq_admit_flag:    # Two thresholds (1 for each demographic group)

            action = self.agents[0].action      # Get action from RL agent    
            action_w = action[:, :1]
            action_nw = action[:, 1:2]
            race_w = df_sample['race']==0
            race_nw = df_sample['race']==1
            race_w = np.expand_dims(race_w,1)
            race_nw = np.expand_dims(race_nw,1)
            decisions = (((scores>action_w) & race_w) |  ((scores>action_nw) & race_nw)).flatten()
            
        else:                                   # One global threshold
            action = self.agents[0].action  # Get action from RL agent
                
            decisions = (scores>action).flatten()        

        # Tallies stats

        # Static
        N_Applied_static = len(scores)                                             # Tally total number of applicants in current time step
        N_Applied_w_static = len(scores[df_sample['race']==0] )               # Tally total number of white applicants in current time step
        N_Applied_nw_static = len(scores[df_sample['race']==1] )              # Tally total number of non-white applicants in current time step
        N_Accepted_static = sum(decisions)                                         # Tally total number of ACCEPTED applicants in current time step
        N_Accepted_w_static = sum((decisions&(df_sample['race']==0)))         # Tally total number of white ACCEPTED applicants in current time step
        N_Accepted_nw_static = sum((decisions&(df_sample['race']==1)))        # Tally total number of non-white ACCEPTED applicants in current time step
        N_Score_Sum_static = float(sum(scores))                                    # Sum of scores of all applicants applying in this round
        N_Score_Sum_w_static = float(sum(scores[df_sample['race']==0] ))      # Sum of scores of all white applicants applying in this round
        N_Score_Sum_nw_static = float(sum(scores[df_sample['race']==1] ))     # Sum of scores of all non-white applicants applying in this round

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

        #---------DISBURSE actions-----------
        action      = self.agents[1].action
        disb_actions_admitted = action[decisions]

        if len(disb_actions_admitted) > 0:

            waitTime2   = disb_actions_admitted<np.percentile(disb_actions_admitted, 20)
            waitTime4   = np.logical_and(disb_actions_admitted>=np.percentile(disb_actions_admitted, 20), disb_actions_admitted<np.percentile(disb_actions_admitted, 40))
            waitTime6   = np.logical_and(disb_actions_admitted>=np.percentile(disb_actions_admitted, 40), disb_actions_admitted<np.percentile(disb_actions_admitted, 60))
            waitTime8   = np.logical_and(disb_actions_admitted>=np.percentile(disb_actions_admitted, 60), disb_actions_admitted<np.percentile(disb_actions_admitted, 80))
            waitTime10  = disb_actions_admitted>=np.percentile(disb_actions_admitted, 80)
            waitTime    = np.zeros(disb_actions_admitted.shape)
            waitTime[waitTime2] = 0.0
            waitTime[waitTime4] = 0.25
            waitTime[waitTime6] = 0.5
            waitTime[waitTime8] = 0.75
            waitTime[waitTime10]= 1
        else:
            waitTime    = np.zeros(disb_actions_admitted.shape)
            
        N_Disbursed_static      = N_Accepted_static                 # Tally total number of ACCEPTED applicants in current time step
        N_Disbursed_w_static    = N_Accepted_w_static               # Tally total number of white ACCEPTED applicants in current time step
        N_Disbursed_nw_static   = N_Accepted_nw_static              # Tally total number of non-white ACCEPTED applicants in current time step

        # Update total cumulative disbursements
        N_Disbursed += N_Disbursed_static
        N_Disbursed_w += N_Disbursed_w_static
        N_Disbursed_nw += N_Disbursed_nw_static

        # Update Wait Times
        WaitTimes_w_static  = sum(waitTime[df_sample['race'][decisions]==0].flatten()) # JUST LEFT OFF CHECKING THIS!!!
        WaitTimes_nw_static = sum(waitTime[df_sample['race'][decisions]==1].flatten())
        WaitTimes    += WaitTimes_w_static + WaitTimes_nw_static
        WaitTimes_w  += WaitTimes_w_static
        WaitTimes_nw += WaitTimes_nw_static

        #---------DEBT actions-----------
        # Apply debt forgiveness actions and update totals
        df_sample_admitted      = df_sample[decisions]
        X_sample_adm_admitted   = X_sample_adm[decisions]
        X_sample_prop_admitted  = X_sample_prop[decisions]
        scores_admitted         = scores[decisions]
        loan_amnt               = np.expand_dims(df_sample_admitted['loan_amnt'].to_numpy(), 1)
        installment             = np.expand_dims(df_sample_admitted['installment'].to_numpy(), 1)
        int_rate                = np.expand_dims(df_sample_admitted['int_rate'].to_numpy(), 1)
        
        # Forgive debt
        if AwareOrNot and not eq_debt_flag:       # different debt forgivness by sensitive attribute

            action = self.agents[2].action
            first_idx = np.array([True,False])
            second_idx = np.array([False,True])
            loan_amnt_forgiven = np.zeros(loan_amnt.shape)
            loan_amnt_forgiven[df_sample_admitted['race']==0] = action[:, first_idx]*loan_amnt[df_sample_admitted['race']==0]
            loan_amnt_forgiven[df_sample_admitted['race']==1] = action[:, second_idx]*loan_amnt[df_sample_admitted['race']==1]

            install_forgiven = np.zeros(loan_amnt.shape)
            install_forgiven[df_sample_admitted['race']==0] = action[:, first_idx]*installment[df_sample_admitted['race']==0]
            install_forgiven[df_sample_admitted['race']==1] = action[:, second_idx]*installment[df_sample_admitted['race']==1]

        else: # same debt forgivness by sensitive attribute

            action = self.agents[2].action
            loan_amnt_forgiven = action*loan_amnt_total
            install_forgiven = action*install_total
            
        # Adjust for forgiveness
        loan_amnt_remain = loan_amnt-loan_amnt_forgiven
        loan_amnt_remain[loan_amnt_remain<0] = 0

        install_remain = installment-install_forgiven
        install_remain[install_remain<0] = 0

        # Customer payments
        Y = np.minimum(install_remain, (scores_admitted*installment))      # Amount paid by customer
        Y_total = install_forgiven + Y                          # Total installment amount (including debt forgiven on installment)
        repaidIdx = Y_total > 0.8*installment

        # Update bank earnings
        Profits = Profits_prev
        Profits += sum((repaidIdx.flatten()*loan_amnt_remain.flatten()*(1+int_rate).flatten()))
        Profits += sum((np.abs(1-repaidIdx).flatten()*0.95*loan_amnt_remain.flatten()))        #  recover_collateral
        Profits -= sum((loan_amnt.flatten()*1.04))

        # Tally defaults, repaid loans

        # Static
        defaultIdx = repaidIdx==0
        N_Defaults_static = sum(defaultIdx)
        N_Defaults_w_static = sum( np.logical_and(defaultIdx.flatten(), df_sample_admitted['race']==0) )
        N_Defaults_nw_static = sum( np.logical_and(defaultIdx.flatten(), df_sample_admitted['race']==1) )
        N_Repaid_static = sum(repaidIdx.flatten())
        N_Repaid_w_static = sum(np.logical_and(repaidIdx.flatten(),(df_sample_admitted['race']==0)))
        N_Repaid_nw_static = sum(np.logical_and(repaidIdx.flatten(),(df_sample_admitted['race']==1)))
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

        #---------Env update-----------
        MarginalProfits = Profits - Profits_prev
        
        # -----------Reward calculation----------
        reward = np.array([[MarginalProfits, \
                            N_Applied_w_static, N_Applied_nw_static, N_Accepted_w_static, N_Accepted_nw_static, \
                            WaitTimes_w_static, WaitTimes_nw_static, N_Disbursed_w_static, N_Disbursed_nw_static, \
                            N_Defaults_w_static, N_Defaults_nw_static, N_Score_Sum_w_static, N_Score_Sum_nw_static, \
                            N_Terminated_w_static, N_Terminated_nw_static]])

        # Wait times
        dw_t = np.abs(WaitTimes_w/(N_Disbursed_w+1e-10) - WaitTimes_nw/(N_Disbursed_nw+1e-10))

        # Update features in feature pool (the dataframe)
        df_pop, X_pop= self.feature_updates(df_pop, df_sample_admitted, X_pop, X_sample_adm_admitted,\
                                                X_sample_prop_admitted, defaultIdx, repaidIdx, norm_params)

        # Default rate disparity
        df_t = np.abs((N_Defaults_w+1e-10)/(N_Terminated_w+1e-10) - (N_Defaults_nw+1e-10)/(N_Terminated_nw+1e-10)) # Default rate disparity

        # Score disparity (proxy for qualification rate)
        ds_t = np.abs((N_Score_Sum_w+1e-10)/(N_Applied_w+1e-10) - (N_Score_Sum_nw+1e-10)/(N_Applied_nw+1e-10))

        # Admissions rate disparity
        da_t = np.abs((N_Accepted_w+1e-10)/(N_Applied_w+1e-10) - (N_Accepted_nw+1e-10)/(N_Applied_nw+1e-10))

        full_reward = np.array([[Profits, ds_t, df_t, dw_t, da_t]]) # These are cumulative, not isolated (we don't use them for model training but instead for indicator tracking.

        bank_default_flag = False
        
        # Update variables (adm)
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

        # Update variables (disburse)
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

        # Update variables (debt)
        self.trackingindicators.N_Defaults = N_Defaults
        self.trackingindicators.N_Defaults_w = N_Defaults_w
        self.trackingindicators.N_Defaults_nw = N_Defaults_nw
        self.trackingindicators.N_Repaid = N_Repaid
        self.trackingindicators.N_Repaid_w = N_Repaid_w
        self.trackingindicators.N_Repaid_nw = N_Repaid_nw
        self.trackingindicators.N_Terminated_w = N_Terminated_w
        self.trackingindicators.N_Terminated_nw = N_Terminated_nw

        # Environment updates
        self.allcustomerinformation.df_pop = df_pop
        self.allcustomerinformation.X_pop = X_pop
        self.trackingindicators.Profits = Profits
        self.trackingindicators.reward = reward
        self.trackingindicators.full_reward = full_reward
        self.allbankinformation.bank_default_flag = bank_default_flag


    # Choose to admit new people into the loan system
    def update_agent_states(self):

        # nn: number samples selected from the dataset

        X_pop = self.allcustomerinformation.X_pop
        linPop = self.allcustomerinformation.linPop
        df_pop = self.allcustomerinformation.df_pop
        W_adm = self.envmodelparameters.W_adm
        b_adm = self.envmodelparameters.b_adm
        X_sample_prop = self.allcustomerinformation.X_sample_prop
        X_sample_prop_curr = self.allcustomerinformation.X_sample_prop_curr
        full_reward = self.trackingindicators.full_reward
        Profits = self.trackingindicators.Profits
        nn = self.nn
        NPerPeriod = self.NPerPeriod
        AwareOrNot = self.AwareOrNot

        # Sample subset of population data
        Pop_sample_idxs = np.array(random.sample(linPop, nn)).astype('int')
        X_sample_adm = X_pop['adm'][Pop_sample_idxs]
        X_sample_prop = X_pop['prop'][Pop_sample_idxs]
        df_sample = df_pop.iloc[Pop_sample_idxs]
            
        # Scores of sample applicants
        if AwareOrNot: # Aware
            scores = self.logistic(W_adm, b_adm, X_sample_adm)
        else: # Unaware
            scores  = self.logistic(W_adm, b_adm, X_sample_adm[:,:-1])
            
        # State variable calculation
        M, N = X_sample_adm.shape
        augment = np.zeros((M,4))
        augment[:,0] = scores.flatten()
        full_reward_copy = full_reward.copy()
        full_reward_copy[0,0] = Profits/(3e7)# np.log(1+Profits)

        augment[:,1:4] = np.repeat(full_reward_copy[:,:3],M,0)

        X_sample_adm_aug = np.hstack(( X_sample_adm, augment))
            
        # If unaware of sensitive attribute, remove it as a feature
        if not AwareOrNot: 
            X_sample_adm_aug = np.delete(X_sample_adm_aug, 9, 1)

        # Update all agent states
        self.agents[0].state = X_sample_adm_aug
        self.agents[1].state = X_sample_adm_aug
        self.agents[2].state = X_sample_adm_aug
        self.allcustomerinformation.df_sample      = df_sample
        self.allcustomerinformation.X_sample_adm   = X_sample_adm
        self.allcustomerinformation.X_sample_prop  = X_sample_prop
        self.allcustomerinformation.scores = scores

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



    def feature_updates(self, df_pop, df_sample, X_pop, X_sample_adm, X_sample_prop, defaultIdx, repaidIdx, norm_params):

        # Heuristic method for updating features given you did or did not repay your loans

        # Dataframe population updates based on terminated loans
        # Fico score for defaulted loans drop by 50. Delinquincies in past 2 years increases.
        defaultedLocs = df_sample.loc[defaultIdx].index
        repaidLocs = df_sample.loc[repaidIdx].index
        
        df_pop.loc[np.in1d(df_pop.index,defaultedLocs), "fico_range_low"]-=100
        df_pop.loc[np.in1d(df_pop.index,defaultedLocs), "fico_range_high"]-=100
        df_pop.loc[np.in1d(df_pop.index,defaultedLocs), "mths_since_last_delinq"]-=5

        X_pop["adm"][np.in1d(df_pop.index,defaultedLocs), 3] = (np.minimum(np.maximum(X_pop["adm"][np.in1d(df_pop.index,defaultedLocs), 3]*norm_params["sig_test"][3]-5+norm_params["mu_test"][3],0),25)-norm_params["mu_test"][3])/norm_params["sig_test"][3]
        X_pop["adm"][np.in1d(df_pop.index,defaultedLocs), 4] = (np.minimum(np.maximum(X_pop["adm"][np.in1d(df_pop.index,defaultedLocs), 4]*norm_params["sig_test"][4]-100+norm_params["mu_test"][4],300),850)-norm_params["mu_test"][4])/norm_params["sig_test"][4]
        X_pop["adm"][np.in1d(df_pop.index,defaultedLocs), 5] = (np.minimum(np.maximum(X_pop["adm"][np.in1d(df_pop.index,defaultedLocs), 5]*norm_params["sig_test"][5]-100+norm_params["mu_test"][5],300),850)-norm_params["mu_test"][5])/norm_params["sig_test"][5]

        X_pop["prop"][np.in1d(df_pop.index,defaultedLocs), 3] -= 5
        X_pop["prop"][np.in1d(df_pop.index,defaultedLocs), 4] -= 100
        X_pop["prop"][np.in1d(df_pop.index,defaultedLocs), 5] -= 100
    
        # Fico score for repaid loans improve by 50. Delinquincies in past 2 years decrease.
        df_pop.loc[np.in1d(df_pop.index,repaidLocs), "fico_range_low"]+=100
        df_pop.loc[np.in1d(df_pop.index,repaidLocs), "fico_range_high"]+=100
        df_pop.loc[np.in1d(df_pop.index,repaidLocs), "mths_since_last_delinq"]+=5

        X_pop["adm"][np.in1d(df_pop.index,repaidLocs), 3] = (np.minimum(np.maximum(X_pop["adm"][np.in1d(df_pop.index,repaidLocs), 3]*norm_params["sig_test"][3]+5+norm_params["mu_test"][3],0),25)-norm_params["mu_test"][3])/norm_params["sig_test"][3]
        X_pop["adm"][np.in1d(df_pop.index,repaidLocs), 4] = (np.minimum(np.maximum(X_pop["adm"][np.in1d(df_pop.index,repaidLocs), 4]*norm_params["sig_test"][4]+100+norm_params["mu_test"][4],300),850)-norm_params["mu_test"][4])/norm_params["sig_test"][4]
        X_pop["adm"][np.in1d(df_pop.index,repaidLocs), 5] = (np.minimum(np.maximum(X_pop["adm"][np.in1d(df_pop.index,repaidLocs), 5]*norm_params["sig_test"][5]+100+norm_params["mu_test"][5],300),850)-norm_params["mu_test"][5])/norm_params["sig_test"][5]

        X_pop["prop"][np.in1d(df_pop.index,repaidLocs), 3] += 5
        X_pop["prop"][np.in1d(df_pop.index,repaidLocs), 4] += 100
        X_pop["prop"][np.in1d(df_pop.index,repaidLocs), 5] += 100

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
    
        return df_pop, X_pop
