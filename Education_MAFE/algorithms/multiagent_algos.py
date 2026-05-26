import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from torch.autograd import Variable
import numpy as np
import pdb
from matplotlib import pyplot as plt

MSELoss = torch.nn.MSELoss()

class CrossEntropyMethod(object):
    """
    Wrapper class for training cross entropy method
    """
    def __init__(self,
                 env,
                 policyTypes,
                 lmds               = [1]*10,
                 save_path          = [],
                 p_elite            = 0.2,
                 tracking_path      = None,
                 opt_type           = "independent",
                 which_agents       = [True]*4,
                 N_epochs           = 50,
                 N_episodes         = 100,
                 episode_length     = 100,
                 ins_mu_scale       = 0,
                 ins_var_scale      = 1,
                 ins_mu_scale_b     = 0,
                 ins_var_scale_b    = 1,
                 hosp_mu_scale      = 0,
                 hosp_var_scale     = 1,
                 hosp_mu_scale_b    = 0,
                 hosp_var_scale_b   = 1,
                 planner_mu_scale   = 0,
                 planner_var_scale  = 1,
                 planner_mu_scale_b = 0,
                 planner_var_scale_b= 1,
                 N_geog             = 9): 

        # Layer information (adjacent values make up dimensions of a layer. Thus, N entries mean N-1 hidden layers (plus postential last output))
        N_feat = {"univ_adm":env.world.agents[0].state.shape[1], "univ_budget":env.world.agents[1].state.shape[1],
                  "emp":env.world.agents[2].state.shape[1], "planner":env.world.agents[3].state.shape[1]}    # Additions at the end based on augmented features comes from the augmented features from the simulation
        intervene_Struct = []

        # University admissions agent policy type
        if policyTypes[0] == 1:
            layer_szs_univ_adm   = [N_feat["univ_adm"],  N_feat["univ_adm"]+1]
            intervene_Struct.append(1)             
        else:
            layer_szs_univ_adm = None
            intervene_Struct.append(None)

        # University budget agent policy type
        if policyTypes[1] == 1:
            layer_szs_univ_budget  = [N_feat["univ_budget"],6, 6] 
            intervene_Struct.append(1)
        else:
            layer_szs_univ_budget = None
            intervene_Struct.append(None)

        # Employer agent policy type
        if policyTypes[2] == 1:
            layer_szs_emp = [N_feat["emp"],  N_feat["emp"]+1]         
            intervene_Struct.append(1)
        else:
            layer_szs_emp  = None
            intervene_Struct.append(None)

        # Planner agent policy type
        if policyTypes[3] == 1:
            layer_szs_planner = [N_feat["planner"], 3+N_geog, 3+N_geog]         
            intervene_Struct.append(1)
        else:
            layer_szs_planner  = None
            intervene_Struct.append(None)

        # Combine layer size parameters in list
        layer_szs               = [layer_szs_univ_adm, layer_szs_univ_budget, layer_szs_emp, layer_szs_planner]
        self.agents             = ["univ_adm", "univ_budget", "emp", "planner"]
        self.obs                = None
        self.actions            = [None, None, None, None]
        self.epoch              = 0
        self.layer_szs          = layer_szs
        self.p_elite            = p_elite
        self.mean_elite_rewards = -np.inf
        self.intervene_Struct   = intervene_Struct
        self.lmds               = lmds
        self.policyTypes        = policyTypes
        self.opt_type           = opt_type
        self.tracking_path      = tracking_path
        self.N_episodes         = N_episodes
        self.dist_agents        = self.construct_intervention_dictionary(layer_szs, True)
        self.LossTrackers       = self.constructLossTrackers(N_episodes)
        action_sizes            = [None, layer_szs_univ_budget[-1], 2, layer_szs_planner[-1]]
        self.ActionTrackers     = self.constructActionTrackers(action_sizes, N_epochs, N_episodes, episode_length-1)
        self.which_agents       = which_agents
        self.scale_consts       = {"univ_adm":None, "univ_budget":None, "emp":209515.0945172529, "planner":None}

    def reset_epoch(self):

        # Set up trackers for learning models
        self.ModelTrackers = self.constructModelTrackers()
        self.DictRealizations_N = []

    def reset_episode(self):

        self.reward_t = np.zeros((0,21))
        self.AgentRealization = None

    def update_rewards(self):

        ModelTrackers       = self.ModelTrackers

        # Segement relvant information from reward_t array
        Demographic_applied_univ        = self.reward_t[:,0:2]
        Demographic_admitted_univ       = self.reward_t[:,2:4]
        Demographic_went_to_undergrad   = self.reward_t[:,4:6]
        Demographic_graduated_undergrad = self.reward_t[:,6:8]
        Demographic_went_to_masters     = self.reward_t[:,8:10]
        Demographic_graduated_masters   = self.reward_t[:,10:12]
        Demographic_went_to_doctorate   = self.reward_t[:,12:14]
        Demographic_graduated_doctorate = self.reward_t[:,14:16]
        emp_profits                     = self.reward_t[:,16]
        Demographic_in_workforce        = self.reward_t[:,17:19]
        Demographic_salary              = self.reward_t[:,19:21]
            
        eps=1e-10

        # Salary Equity
        ave_salaries        = np.sum(Demographic_salary,axis=0)/np.sum(Demographic_in_workforce,axis=0)
        salary_Eqty         = -np.abs(ave_salaries[0]-ave_salaries[1])/(ave_salaries[0]+ave_salaries[1]+eps)

        # Admissions Equity
        adm_Rts             = np.sum(Demographic_admitted_univ,axis=0)/(np.sum(Demographic_applied_univ,axis=0)+eps)
        adm_Eqty            = -np.abs(adm_Rts[0]-adm_Rts[1])

        # Graduation Rate Equity (Undergrad)
        grad_undergrad      = np.sum(Demographic_graduated_undergrad[5:,:], axis=0)/(np.sum(Demographic_went_to_undergrad[2:-3,:], axis=0) + eps)
        grad_undergrad_Eqty = -np.abs(grad_undergrad[0] - grad_undergrad[1])

        # Graduation Rate Equity (Masters)
        grad_masters        = np.sum(Demographic_graduated_masters[7:,:], axis=0)/(np.sum(Demographic_went_to_masters[6:-1,:], axis=0)+eps)
        grad_masters_Eqty   = -np.abs(grad_masters[0] - grad_masters[1])

        # Graduation Rate Equity (Ph.D)
        grad_doctorate      = np.sum(Demographic_graduated_doctorate[12:,:], axis=0)/(np.sum(Demographic_went_to_doctorate[8:-4,:], axis=0)+eps)
        grad_doctorate_Eqty = -np.abs(grad_doctorate[0] - grad_doctorate[1])

        # Total Graduation Rates
        grad_undergrad_Rt   = np.sum(Demographic_graduated_undergrad[5:,:])/(np.sum(Demographic_went_to_undergrad[2:-3,:])+eps)
        grad_masters_Rt     = np.sum(Demographic_graduated_masters[7:,:])/(np.sum(Demographic_went_to_masters[6:-1,:])+eps)
        grad_doctorate_Rt   = np.sum(Demographic_graduated_doctorate[12:,:])/(np.sum(Demographic_went_to_doctorate[8:-4,:])+eps)

        # Total Admissions Rates
        adm_Rts             = np.sum(Demographic_admitted_univ)/(np.sum(Demographic_applied_univ)+eps)

        # Employer Profits
        emp_profits = np.sum(emp_profits)

        # Store model performance information for constructing loss values
        ModelTrackers["salary_Eqty_arr"]            = np.append(ModelTrackers["salary_Eqty_arr"], salary_Eqty)
        ModelTrackers["adm_Eqty_arr"]               = np.append(ModelTrackers["adm_Eqty_arr"], adm_Eqty)
        ModelTrackers["adm_Rts_arr"]                = np.append(ModelTrackers["adm_Rts_arr"], adm_Rts)
        ModelTrackers["grad_undergrad_Eqty_arr"]    = np.append(ModelTrackers["grad_undergrad_Eqty_arr"], grad_undergrad_Eqty)
        ModelTrackers["grad_masters_Eqty_arr"]      = np.append(ModelTrackers["grad_masters_Eqty_arr"], grad_masters_Eqty)
        ModelTrackers["grad_doctorate_Eqty_arr"]    = np.append(ModelTrackers["grad_doctorate_Eqty_arr"], grad_doctorate_Eqty)
        ModelTrackers["grad_undergrad_Rt_arr"]      = np.append(ModelTrackers["grad_undergrad_Rt_arr"], grad_undergrad_Rt)
        ModelTrackers["grad_masters_Rt_arr"]        = np.append(ModelTrackers["grad_masters_Rt_arr"], grad_masters_Rt)
        ModelTrackers["grad_doctorate_Rt_arr"]      = np.append(ModelTrackers["grad_doctorate_Rt_arr"], grad_doctorate_Rt)
        ModelTrackers["emp_profits_arr"]            = np.append(ModelTrackers["emp_profits_arr"], emp_profits)

        self.ModelTrackers = ModelTrackers



    def construct_intervention_dictionary(self, layer_szs, train=True, save_path = [],
                                      univ_adm_mu_scale=0, univ_adm_var_scale=1, univ_adm_mu_scale_b=0, univ_adm_var_scale_b=1,
                                      univ_budget_mu_scale=0, univ_budget_var_scale=1, univ_budget_mu_scale_b=0, univ_budget_var_scale_b=1,
                                      emp_mu_scale=0, emp_var_scale=1, emp_mu_scale_b=0, emp_var_scale_b=1,
                                      planner_mu_scale=0, planner_var_scale=1, planner_mu_scale_b=0, planner_var_scale_b=1):

        # Create distribution parameters used to sample weights of the model

        # Import pretrained weights
        if len(save_path) > 0:
            dist_agents = np.load(save_path, allow_pickle=True).item()
            return dist_agents

        # Create distribution parameters used to sample weights of the model
        AgentNames = ['univ_adm', 'univ_budget', 'emp', 'planner']
        dist_agents = {'univ_adm':None, 'univ_budget':None, 'emp':None, 'planner':None}

        # Get layer sizes for each model
        layer_szs_univ_adm, layer_szs_univ_budget, layer_szs_emp, layer_szs_planner = layer_szs

        if not train:
            # Perconstructed InterveneDict goes here
            return InterveneDict

        # Initialize agent storage
        for ii in range(4):
            dist_agents[AgentNames[ii]] = {}
        
        N_layers_univ_adm       = len(layer_szs[0])
        N_layers_univ_budget    = len(layer_szs[1])-1
        N_layers_univ_emp       = len(layer_szs[2])
        N_layers_planner        = len(layer_szs[3])-1

        # Create admissions score weights
        for ii in range(N_layers_univ_adm):
            dist_agents['univ_adm']['univ_adm_mu'+str(ii)] = np.ones((layer_szs_univ_adm[ii], 1))*univ_adm_mu_scale    # Mean of weights dist
            dist_agents['univ_adm']['univ_adm_var'+str(ii)] = np.ones((layer_szs_univ_adm[ii], 1))*univ_adm_var_scale  # Std of weights dist
            dist_agents['univ_adm']['univ_adm_mu_b'+str(ii)] = np.ones((1,1))*univ_adm_mu_scale_b                  # Mean of bias dist
            dist_agents['univ_adm']['univ_adm_var_b'+str(ii)] = np.ones((1,1))*univ_adm_var_scale_b                # Std of bias dist

        # Create university program weights
        for ii in range(N_layers_univ_budget):
            dist_agents['univ_budget']['univ_budget_mu'+str(ii)] = np.ones((layer_szs_univ_budget[ii], layer_szs_univ_budget[ii+1]))*univ_budget_mu_scale      # Mean of weights dist
            dist_agents['univ_budget']['univ_budget_var'+str(ii)] = np.ones((layer_szs_univ_budget[ii], layer_szs_univ_budget[ii+1]))*univ_budget_var_scale    # Std of weights dist
            dist_agents['univ_budget']['univ_budget_mu_b'+str(ii)] = np.ones((1,1))*univ_budget_mu_scale_b                                             # Mean of bias dist
            dist_agents['univ_budget']['univ_budget_var_b'+str(ii)] = np.ones((1,1))*univ_budget_var_scale_b                                           # Std of bias dist

        # Create hospital bed weights
        for ii in range(N_layers_univ_emp):
            dist_agents['emp']['emp_mu'+str(ii)]      = np.ones((layer_szs_emp[ii], 1))*emp_mu_scale    # Mean of weights dist
            dist_agents['emp']['emp_var'+str(ii)]     = np.ones((layer_szs_emp[ii], 1))*emp_var_scale  # Std of weights dist
            dist_agents['emp']['emp_mu_b'+str(ii)]    = np.ones((1,1))*emp_mu_scale_b                  # Mean of bias dist
            dist_agents['emp']['emp_var_b'+str(ii)]   = np.ones((1,1))*emp_var_scale_b                # Std of bias dist

        # Create planner investment weights
        for ii in range(N_layers_planner):

            dist_agents['planner']['planner_mu'+str(ii)] = np.ones((layer_szs_planner[ii], layer_szs_planner[ii+1]))*planner_mu_scale      # Mean of weights dist
            dist_agents['planner']['planner_var'+str(ii)] = np.ones((layer_szs_planner[ii], layer_szs_planner[ii+1]))*planner_var_scale    # Std of weights dist
            dist_agents['planner']['planner_mu_b'+str(ii)] = np.ones((1,1))*planner_mu_scale_b                                             # Mean of bias dist
            dist_agents['planner']['planner_var_b'+str(ii)] = np.ones((1,1))*planner_var_scale_b                                           # Std of bias dist

        return dist_agents

    def sample_intervention_parameters(self):

        # Sample the weights of the current model.
        AgentNames = ['univ_adm', 'univ_budget', 'emp', 'planner']
        self.AgentRealization = {'univ_adm':[], 'univ_budget':[], 'emp':[], 'planner':[]}
        for ii in range(4):

            if self.dist_agents[AgentNames[ii]] is None:
                continue

            CurrDict = self.dist_agents[AgentNames[ii]]        # current dictionary used for sampling weights
            CurrDict_store  = {}                            # current dictionary used for storing sampled weights

            N = int(len(CurrDict)/4)                        # Number of layers in current model

            for jj in range(N):

                # Size of weight params for this layer
                CurrWeights_mu  = CurrDict[AgentNames[ii] + "_mu" + str(jj)]
                CurrWeights_mu_b   = CurrDict[AgentNames[ii] + "_mu_b" + str(jj)]
                CurrWeights_var = CurrDict[AgentNames[ii] + "_var" + str(jj)]
                CurrWeights_var_b = CurrDict[AgentNames[ii] + "_var_b" + str(jj)]

                # Convert variance to standard deviation
                CurrWeights_sig = np.sqrt(CurrWeights_var)
                CurrWeights_sig_b = np.sqrt(CurrWeights_var_b)

                # Sample weights from Gaussian
                Params_w = np.random.normal(CurrWeights_mu, CurrWeights_sig)
                Params_b = np.random.normal(CurrWeights_mu_b, CurrWeights_sig_b)

                # Store weights for current agent
                CurrDict_store[AgentNames[ii]+"_w"+str(jj)] = Params_w
                CurrDict_store[AgentNames[ii]+"_b"+str(jj)] = Params_b

            # Store each agent's weights in one object
            self.AgentRealization[AgentNames[ii]] = CurrDict_store    

    def GetActions(self, obs):

        # Input:  features
        # Output: premium

        actions = []
        for nn in range(len(self.agents)):
            
            agent = self.agents[nn]
        
            # Load data used by agent to make decision
            X_agent= obs[nn]

            if self.which_agents[nn]:

                # Get actions (multiple of this agent)
                if X_agent is None or X_agent.shape[0] == 0:
                    action = None
                else:
                    action = self.apply_intervention_model(X_agent, agent, self.scale_consts[agent])
                
            actions.append(action)

        return actions

    def apply_intervention_model(self, X, agent, scale_const=None):

        # Given module that we want to get result for and the associated data, we apply gather appropriate weights from Model_Dict for this module and apply then to the data

        # Get current module's weights
        CurrDict = self.AgentRealization[agent]

        # Apply the weights appropriately
        if agent == 'univ_adm':

            # init params
            M_batch, N_feat = X.shape
            X_orig = X.copy()

            # Run data through beginning model layers
            N = int(len(CurrDict)/2)        # Divide by 2 since bias and weights each added for one layer, doubling its size
            for ii in range(N-1):
                X = self.sig( X@CurrDict[agent+"_w"+str(ii)] + CurrDict[agent+"_b"+str(ii)] )

            # Average output and run through final layer
            X = np.mean(X,axis=0)

            # Append global feature to data and run through final MLP layer
            X = np.hstack((X_orig, np.ones((M_batch,1))*X))
            ii += 1
            X = self.sig( X@CurrDict[agent+"_w"+str(ii)] + CurrDict[agent+"_b"+str(ii)] )
            adm_scores = X

            return adm_scores.flatten()


        # Apply the weights appropriately
        if agent == 'univ_budget':

            # Run data through beginning model layers
            N = int(len(CurrDict)/2)        # Divide by 2 since bias and weights each added for one layer, doubling its size
            for ii in range(N-1):
                X = self.sig( X@CurrDict[agent+"_w"+str(ii)] + CurrDict[agent+"_b"+str(ii)] )

            # Average output and run through final layer
            X = np.expand_dims(np.mean(X,axis=0),axis=0)
            X = X@CurrDict[agent+"_w"+str(N-1)] + CurrDict[agent+"_b"+str(N-1)]
            X = X.flatten()

            # Investment apportioning step
            soft_vals = np.array([np.exp(X[0]), np.exp(X[1]), np.exp(X[2]), np.exp(X[3]), np.exp(X[4]), np.exp(X[5])])

            X = soft_vals/np.sum(soft_vals)

            univ_budget_subs = X

            return univ_budget_subs.flatten()

        # Apply the weights appropriately
        if agent == 'emp':

            # init params
            M_batch, N_feat = X.shape
            X_orig = X.copy()

            # Run data through beginning model layers
            N = int(len(CurrDict)/2)        # Divide by 2 since bias and weights each added for one layer, doubling its size
            for ii in range(N-1):
                X = self.sig( X@CurrDict[agent+"_w"+str(ii)] + CurrDict[agent+"_b"+str(ii)] )

            # Average output and run through final layer
            X = np.mean(X,axis=0)

            # Append global feature to data and run through final MLP layer
            X = np.hstack((X_orig, np.ones((M_batch,1))*X))
            ii += 1
            X = self.sig( X@CurrDict[agent+"_w"+str(ii)] + CurrDict[agent+"_b"+str(ii)] )

            emp_salaries = X*scale_const

            return emp_salaries.flatten()

        # Apply the weights appropriately
        if agent == 'planner':

            # Run data through beginning model layers
            N = int(len(CurrDict)/2)        # Divide by 2 since bias and weights each added for one layer, doubling its size
            for ii in range(N-1):
                X = self.sig( X@CurrDict[agent+"_w"+str(ii)] + CurrDict[agent+"_b"+str(ii)] )

            # Average output and run through final layer
            X = np.expand_dims(np.mean(X,axis=0),axis=0)
            X = X@CurrDict[agent+"_w"+str(N-1)] + CurrDict[agent+"_b"+str(N-1)]
            X = X.flatten()

            # Investment apportioning step
            soft_vals = np.array([np.exp(X[0]), np.exp(X[1]), np.exp(X[2]),
                                  np.exp(X[3]), np.exp(X[4]), np.exp(X[5]), np.exp(X[6]), np.exp(X[7]), np.exp(X[8]), np.exp(X[9]), np.exp(X[10]), np.exp(X[11])])

            X[:3] = soft_vals[:3]/np.sum(soft_vals[:3])
            X[3:12] = soft_vals[3:12]/np.sum(soft_vals[3:12])

            planner_subs = X

            return planner_subs.flatten()


    # Sigmoid function
    def sig(self, x):
        return 1 / (1 + np.exp(-x))
   
    def step(self, weight=0.67):

        tracking_path                   = self.tracking_path
        DictRealizations_N              = self.DictRealizations_N
        emp_profits_rewards             = self.ModelTrackers["emp_profits_arr"]
        salary_minorEqty_rewards        = self.ModelTrackers["salary_Eqty_arr"]
        admRts_minorEqty_rewards        = self.ModelTrackers["adm_Eqty_arr"]
        admRts_rewards                  = self.ModelTrackers["adm_Rts_arr"]
        gradRts_UG_rewards              = self.ModelTrackers["grad_undergrad_Rt_arr"]
        gradRts_MS_rewards              = self.ModelTrackers["grad_masters_Rt_arr"]
        gradRts_PH_rewards              = self.ModelTrackers["grad_doctorate_Rt_arr"]
        gradRts_UG_minorEqty_rewards    = self.ModelTrackers["grad_undergrad_Eqty_arr"]
        gradRts_MS_minorEqty_rewards    = self.ModelTrackers["grad_masters_Eqty_arr"]
        gradRts_PH_minorEqty_rewards    = self.ModelTrackers["grad_doctorate_Eqty_arr"]
        LossTrackers                    = self.LossTrackers
        opt_type                        = self.opt_type
        lmds                            = self.lmds
        layer_szs                       = self.layer_szs
        p_elite                         = self.p_elite 

        # Organize for looping

        # Training using cooperation or independent schemes        
        lmd_emp_profits             = lmds["emp_profits"]
        lmd_salary_minorEqty        = lmds["salary_minorEqty"]
        lmd_admRts_minorEqty        = lmds["admRts_minorEqty"]
        lmd_admRts                  = lmds["admRts"]
        lmd_gradRts_UG              = lmds["gradRts_UG"]
        lmd_gradRts_MS              = lmds["gradRts_MS"]
        lmd_gradRts_PH              = lmds["gradRts_PH"]
        lmd_gradRts_UG_minorEqty    = lmds["gradRts_UG_minorEqty"]
        lmd_gradRts_MS_minorEqty    = lmds["gradRts_MS_minorEqty"]
        lmd_gradRts_PH_minorEqty    = lmds["gradRts_PH_minorEqty"]

        eps=1e-10
        emp_profits_rewards_norm            = (emp_profits_rewards - np.mean(emp_profits_rewards))/(np.std(emp_profits_rewards)+eps)
        salary_minorEqty_rewards_norm       = (salary_minorEqty_rewards - np.mean(salary_minorEqty_rewards))/(np.std(salary_minorEqty_rewards)+eps)
        admRts_minorEqty_rewards_norm       = (admRts_minorEqty_rewards - np.mean(admRts_minorEqty_rewards))/(np.std(admRts_minorEqty_rewards)+eps)
        admRts_rewards_norm                 = (admRts_rewards - np.mean(admRts_rewards))/(np.std(admRts_rewards)+eps)
        gradRts_UG_rewards_norm             = (gradRts_UG_rewards - np.mean(gradRts_UG_rewards))/(np.std(gradRts_UG_rewards)+eps)
        gradRts_MS_rewards_norm             = (gradRts_MS_rewards - np.mean(gradRts_MS_rewards))/(np.std(gradRts_MS_rewards)+eps)
        gradRts_PH_rewards_norm             = (gradRts_PH_rewards - np.mean(gradRts_PH_rewards))/(np.std(gradRts_PH_rewards)+eps)
        gradRts_UG_minorEqty_rewards_norm   = (gradRts_UG_minorEqty_rewards - np.mean(gradRts_UG_minorEqty_rewards))/(np.std(gradRts_UG_minorEqty_rewards)+eps)
        gradRts_MS_minorEqty_rewards_norm   = (gradRts_MS_minorEqty_rewards - np.mean(gradRts_MS_minorEqty_rewards))/(np.std(gradRts_MS_minorEqty_rewards)+eps)
        gradRts_PH_minorEqty_rewards_norm   = (gradRts_PH_minorEqty_rewards - np.mean(gradRts_PH_minorEqty_rewards))/(np.std(gradRts_PH_minorEqty_rewards)+eps)

        coop_rewards     = (lmd_emp_profits*emp_profits_rewards_norm                + lmd_salary_minorEqty*salary_minorEqty_rewards_norm            + \
                        lmd_admRts_minorEqty*admRts_minorEqty_rewards_norm          + lmd_gradRts_UG*gradRts_UG_rewards_norm                        + \
                        lmd_gradRts_MS*gradRts_MS_rewards_norm                      + lmd_gradRts_PH*gradRts_PH_rewards_norm                        + \
                        lmd_gradRts_UG_minorEqty*gradRts_UG_minorEqty_rewards_norm  + lmd_gradRts_MS_minorEqty*gradRts_MS_minorEqty_rewards_norm    +\
                        lmd_gradRts_PH_minorEqty*gradRts_PH_minorEqty_rewards_norm  + lmd_admRts*admRts_rewards_norm)/10

        univ_rewards     = (lmd_gradRts_UG*gradRts_UG_rewards_norm + \
                            lmd_gradRts_MS*gradRts_MS_rewards_norm + \
                            lmd_gradRts_PH*gradRts_PH_rewards_norm )/3

        ind_rewards = [univ_rewards, univ_rewards, emp_profits_rewards_norm, coop_rewards]
        if opt_type == 'independent':
            train_rewards   = ind_rewards.copy()
        else:
            train_rewards   = [coop_rewards, coop_rewards, coop_rewards, coop_rewards]
        
        # Initialize
        names = ['univ_adm', 'univ_budget', 'emp', 'planner']
        N_layers = len(layer_szs[0])
        p_idx = int(p_elite*len(DictRealizations_N))    # Elite index: index at the p% mark

        for n_agent in range(4):

            # Current agent info
            curr_train_rewards  = train_rewards[n_agent]
            curr_name           = names[n_agent]

            # Gather elite dicts
            idx_sort_train      = curr_train_rewards.argsort()[::-1]

            # Perform elite calculations
            DictRealizations_N_sort = np.array(DictRealizations_N)[idx_sort_train]
            DictRealizations_elite_curr = DictRealizations_N_sort[:p_idx]
            N_elite = len(DictRealizations_elite_curr)

            # Create admissions weights
            for ii in range(N_layers):

                # Admit layer dimensions
                M_w,N_w = DictRealizations_elite_curr[0][curr_name][curr_name + '_w'+str(ii)].shape
                M_b,N_b = DictRealizations_elite_curr[0][curr_name][curr_name + '_b'+str(ii)].shape

                # Store layer information for calculating average and variance of elite set
                curr_layer_w = np.zeros((M_w, N_w, 0))
                curr_layer_b = np.zeros((M_b, N_b, 0))
        
                for n_elite in range(N_elite):

                    # For averaging over agent's layers
                    curr_layer_w = np.dstack((curr_layer_w, DictRealizations_elite_curr[n_elite][curr_name][curr_name+'_w'+str(ii)]))
                    curr_layer_b = np.dstack((curr_layer_b, DictRealizations_elite_curr[n_elite][curr_name][curr_name+'_b'+str(ii)]))

                # Mean of dists
                self.dist_agents[ curr_name ][ curr_name+'_mu'+str(ii)   ] = np.mean(curr_layer_w, axis=2)
                self.dist_agents[ curr_name ][ curr_name+'_mu_b'+str(ii) ] = np.mean(curr_layer_b, axis=2)

                # Variance of dists
                self.dist_agents[ curr_name ][ curr_name+'_var'+str(ii)   ] = np.var(curr_layer_w, axis=2)
                self.dist_agents[ curr_name ][ curr_name+'_var_b'+str(ii) ] = np.var(curr_layer_b, axis=2)

        # Store reward information
        names        = ["emp_profits", "salary_minorEqty", "admRts_minorEqty",
                    "gradRts_UG", "gradRts_MS", "gradRts_PH",
                    "gradRts_UG_minorEqty", "gradRts_MS_minorEqty", "gradRts_PH_minorEqty", "admRts"]
        ind_rewards  = [emp_profits_rewards, salary_minorEqty_rewards, admRts_minorEqty_rewards,
                     gradRts_UG_rewards, gradRts_MS_rewards, gradRts_PH_rewards,
                        gradRts_UG_minorEqty_rewards, gradRts_MS_minorEqty_rewards, gradRts_PH_minorEqty_rewards, admRts_rewards]

        for n_reward in range(len(names)):
            
            # Current reward
            curr_ind_rewards    = ind_rewards[n_reward]
            curr_name           = names[n_reward]

            # Store current disparity average rewards
            LossTrackers[curr_name+"_mean_rewards"]              = np.append(LossTrackers[curr_name + "_mean_rewards"], np.mean(curr_ind_rewards))
            LossTrackers[curr_name+"_rewards_arr"]               = np.vstack((LossTrackers[curr_name + "_rewards_arr"], curr_ind_rewards))    

        # Update loss info saved fill at the end of each epoch
        np.save(tracking_path +'LossTrackers.npy', LossTrackers)

        # Save InterveneDict so we can continue training/ load later on
        np.save(tracking_path +"InterveneDict.npy", self.dist_agents)

        # Save actions from the current epoch
        np.save(tracking_path +'ActionTrackers.npy', self.ActionTrackers)

        # Store loss info back in environment 
        self.LossTrackers = LossTrackers


    def constructLossTrackers(self, N_episodes):

        LossTrackers = {
            "emp_profits_rewards_arr"                   :   np.zeros((0,N_episodes)),
            "salary_minorEqty_rewards_arr"              :   np.zeros((0,N_episodes)),
            "admRts_minorEqty_rewards_arr"              :   np.zeros((0,N_episodes)),
            "admRts_rewards_arr"                        :   np.zeros((0,N_episodes)),
            "gradRts_UG_rewards_arr"                    :   np.zeros((0,N_episodes)),
            "gradRts_MS_rewards_arr"                    :   np.zeros((0,N_episodes)),
            "gradRts_PH_rewards_arr"                    :   np.zeros((0,N_episodes)),
            "gradRts_UG_minorEqty_rewards_arr"          :   np.zeros((0,N_episodes)),
            "gradRts_MS_minorEqty_rewards_arr"          :   np.zeros((0,N_episodes)),
            "gradRts_PH_minorEqty_rewards_arr"          :   np.zeros((0,N_episodes)),

            "emp_profits_mean_rewards"                  :   np.zeros((0)),
            "salary_minorEqty_mean_rewards"             :   np.zeros((0)),
            "admRts_minorEqty_mean_rewards"             :   np.zeros((0)),
            "admRts_mean_rewards"                       :   np.zeros((0)),
            "gradRts_UG_mean_rewards"                   :   np.zeros((0)),
            "gradRts_MS_mean_rewards"                   :   np.zeros((0)),
            "gradRts_PH_mean_rewards"                   :   np.zeros((0)),
            "gradRts_UG_minorEqty_mean_rewards"         :   np.zeros((0)),
            "gradRts_MS_minorEqty_mean_rewards"         :   np.zeros((0)),
            "gradRts_PH_minorEqty_mean_rewards"         :   np.zeros((0)),

            }

        return LossTrackers


    def constructModelTrackers(self):

        ModelTrackers = {
            "salary_Eqty_arr"          :   np.array([]),    
            "adm_Eqty_arr"             :   np.array([]),
            "adm_Rts_arr"              :   np.array([]),  
            "grad_undergrad_Eqty_arr"  :   np.array([]), 
            "grad_masters_Eqty_arr"    :   np.array([]),
            "grad_doctorate_Eqty_arr"  :   np.array([]),
            "grad_undergrad_Rt_arr"    :   np.array([]),
            "grad_masters_Rt_arr"      :   np.array([]),
            "grad_doctorate_Rt_arr"    :   np.array([]), 
            "emp_profits_arr"          :   np.array([])
            }

        return ModelTrackers


    def constructActionTrackers(self, action_szs, N_epochs, N_episodes, episode_len):

        ActionTrackers = {}

        if action_szs[0] is not None:
            ActionTrackers["univ_adm_actions"]      = np.zeros((N_epochs, N_episodes, episode_len, action_szs[0]))

        if action_szs[1] is not None:
            ActionTrackers["univ_budget_actions"]   = np.zeros((N_epochs, N_episodes, episode_len, action_szs[1]))

        if action_szs[2] is not None:
            ActionTrackers["emp_actions"]           = np.zeros((N_epochs, N_episodes, episode_len, action_szs[2]))

        if action_szs[3] is not None:
            ActionTrackers["planner_actions"]       = np.zeros((N_epochs, N_episodes, episode_len, action_szs[3]))


        return ActionTrackers


class MAPPO(object): 
    """
    Wrapper class for PPO-esque (i.e. also MAPPO) agents in multi-agent task

    Credit: obtained from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/algorithms/maddpg.py

    Adapted by Author A
    """

    def __init__(self,
                 env,
                 which_agents,
                 policyTypes,
                 lmds=np.array([1,1,1]),
                 gamma=0.95,
                 lr_critic=0.01,
                 lr_policy=0.001,
                 hidden_dim=64,
                 Fair_scheme=0,
                 track_path=None):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
        """

        self.which_agents = which_agents
        self.nagents = len(which_agents)
        self.agents = []
        self.track_path     = track_path
        self.policyTypes = policyTypes
        self.lmds = np.expand_dims(np.array(lmds),1)
        self.gamma = gamma
        self.lr_policy = lr_policy
        self.lr_critic = lr_critic
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.niter = 0
        self.Fair_scheme = Fair_scheme
        action_sizes        = [None, 6, None, 12]
        self.ActionTrackers = self.constructActionTrackers(action_sizes)
        self.ModelTrackers  = self.constructModelTrackers()
        
        # Create agents

        _,num_in_pol1 = env.world.agents[0].state.shape
        num_in_pol1 = num_in_pol1 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION
        num_in_critic1 = num_in_pol1
                
        _,num_in_pol2 = env.world.agents[1].state.shape
        num_in_pol2 = num_in_pol2 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION
        num_in_critic2 = num_in_pol2
            
        _,num_in_pol3 = env.world.agents[2].state.shape
        num_in_pol3 = num_in_pol3 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION
        num_in_critic3 = num_in_pol3

        _,num_in_pol4 = env.world.agents[3].state.shape
        num_in_pol4 = num_in_pol4 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION
        num_in_critic4 = num_in_pol4

        # Admissions
        if which_agents[0]:
            self.agents.append( AdmissionsAgent_MAPPO(num_in_pol1, policyTypes[0],
                                              num_in_critic1, num_in_critic2, num_in_critic3, num_in_critic4,
                                              hidden_dim, which_agents, lr_critic=lr_critic, lr_policy=lr_policy) )
        else:
            self.agents.append(None)

        # Employer
        if which_agents[1]:
            self.agents.append( BudgetAgent_MAPPO(num_in_pol2, policyTypes[1],
                                         num_in_critic1, num_in_critic2, num_in_critic3, num_in_critic4,
                                         hidden_dim, which_agents, lr_critic=lr_critic, lr_policy=lr_policy) )            
        else:
            self.agents.append(None)

        # Budget
        if which_agents[2]:
            self.agents.append( EmployerAgent_MAPPO(num_in_pol3, policyTypes[2],
                                             num_in_critic1, num_in_critic2, num_in_critic3, num_in_critic4,
                                             hidden_dim, which_agents, lr_critic=lr_critic, lr_policy=lr_policy) )
        else:
            self.agents.append(None)

        # Planner
        if which_agents[3]:
            self.agents.append( PlannerAgent_MAPPO(num_in_pol4, policyTypes[3],
                                         num_in_critic1, num_in_critic2, num_in_critic3, num_in_critic4,
                                         hidden_dim, which_agents, lr_critic=lr_critic, lr_policy=lr_policy) )
        else:
            self.agents.append(None)
        
    @property
    def policies(self):
        P = []
        for ii in range(len(self.agents)):
            if self.agents[ii] is not None:
                P.append(self.agents[ii].policy)
            else:
                P.append(None)
        return P

    def reset_episode(self):

        self.reward_t = np.zeros((0,21))
        self.AgentRealization = None

    def update_rewards(self):

        track_path          = self.track_path
        ModelTrackers       = self.ModelTrackers

        # Calculate rewards from component functions
        Demographic_applied_univ        = self.reward_t[:,0:2]
        Demographic_admitted_univ       = self.reward_t[:,2:4]
        Demographic_went_to_undergrad   = self.reward_t[:,4:6]
        Demographic_graduated_undergrad = self.reward_t[:,6:8]
        Demographic_went_to_masters     = self.reward_t[:,8:10]
        Demographic_graduated_masters   = self.reward_t[:,10:12]
        Demographic_went_to_doctorate   = self.reward_t[:,12:14]
        Demographic_graduated_doctorate = self.reward_t[:,14:16]
        emp_profits                     = self.reward_t[:,16]
        Demographic_in_workforce        = self.reward_t[:,17:19]
        Demographic_salary              = self.reward_t[:,19:21]
            
        eps=1e-10

        # Salary Equity
        ave_salaries        = np.sum(Demographic_salary,axis=0)/np.sum(Demographic_in_workforce,axis=0)
        salary_Eqty         = -np.abs(ave_salaries[0]-ave_salaries[1])/(ave_salaries[0]+ave_salaries[1]+eps)

        # Admissions Equity
        adm_Rts             = np.sum(Demographic_admitted_univ,axis=0)/(np.sum(Demographic_applied_univ,axis=0)+eps)
        adm_Eqty            = -np.abs(adm_Rts[0]-adm_Rts[1])

        # Graduation Rate Equity (Undergrad)
        grad_undergrad      = np.sum(Demographic_graduated_undergrad[5:,:], axis=0)/(np.sum(Demographic_went_to_undergrad[2:-3,:], axis=0) + eps)
        grad_undergrad_Eqty = -np.abs(grad_undergrad[0] - grad_undergrad[1])

        # Graduation Rate Equity (Masters)
        grad_masters        = np.sum(Demographic_graduated_masters[7:,:], axis=0)/(np.sum(Demographic_went_to_masters[6:-1,:], axis=0)+eps)
        grad_masters_Eqty   = -np.abs(grad_masters[0] - grad_masters[1])

        # Graduation Rate Equity (Ph.D)
        grad_doctorate      = np.sum(Demographic_graduated_doctorate[12:,:], axis=0)/(np.sum(Demographic_went_to_doctorate[8:-4,:], axis=0)+eps)
        grad_doctorate_Eqty = -np.abs(grad_doctorate[0] - grad_doctorate[1])

        # Total Graduation Rates
        grad_undergrad_Rt   = np.sum(Demographic_graduated_undergrad[5:,:])/(np.sum(Demographic_went_to_undergrad[2:-3,:])+eps)
        grad_masters_Rt     = np.sum(Demographic_graduated_masters[7:,:])/(np.sum(Demographic_went_to_masters[6:-1,:])+eps)
        grad_doctorate_Rt   = np.sum(Demographic_graduated_doctorate[12:,:])/(np.sum(Demographic_went_to_doctorate[8:-4,:])+eps)

        # Total Admissions Rates
        adm_Rts             = np.sum(Demographic_admitted_univ)/(np.sum(Demographic_applied_univ)+eps)

        # Employer Profits
        emp_profits = np.sum(emp_profits)

        # Store model performance information for constructing loss values
        ModelTrackers["salary_Eqty_arr"]            = np.append(ModelTrackers["salary_Eqty_arr"], salary_Eqty)
        ModelTrackers["adm_Eqty_arr"]               = np.append(ModelTrackers["adm_Eqty_arr"], adm_Eqty)
        ModelTrackers["adm_Rts_arr"]                = np.append(ModelTrackers["adm_Rts_arr"], adm_Rts)
        ModelTrackers["grad_undergrad_Eqty_arr"]    = np.append(ModelTrackers["grad_undergrad_Eqty_arr"], grad_undergrad_Eqty)
        ModelTrackers["grad_masters_Eqty_arr"]      = np.append(ModelTrackers["grad_masters_Eqty_arr"], grad_masters_Eqty)
        ModelTrackers["grad_doctorate_Eqty_arr"]    = np.append(ModelTrackers["grad_doctorate_Eqty_arr"], grad_doctorate_Eqty)
        ModelTrackers["grad_undergrad_Rt_arr"]      = np.append(ModelTrackers["grad_undergrad_Rt_arr"], grad_undergrad_Rt)
        ModelTrackers["grad_masters_Rt_arr"]        = np.append(ModelTrackers["grad_masters_Rt_arr"], grad_masters_Rt)
        ModelTrackers["grad_doctorate_Rt_arr"]      = np.append(ModelTrackers["grad_doctorate_Rt_arr"], grad_doctorate_Rt)
        ModelTrackers["emp_profits_arr"]            = np.append(ModelTrackers["emp_profits_arr"], emp_profits)
        ModelTrackers["policy_univ_adm_loss_arr"]   = self.agents[0].policy_loss
        ModelTrackers["policy_univ_budget_loss_arr"]= self.agents[1].policy_loss
        ModelTrackers["policy_emp_loss_arr"]        = self.agents[2].policy_loss
        ModelTrackers["policy_planner_loss_arr"]    = self.agents[3].policy_loss
        ModelTrackers["value_univ_adm_loss_arr"]    = self.agents[0].critic_loss['Main']
        ModelTrackers["value_univ_budget_loss_arr"] = self.agents[1].critic_loss['Main']
        ModelTrackers["value_emp_loss_arr"]         = self.agents[2].critic_loss['Main']
        ModelTrackers["value_emp_loss_arr"]         = self.agents[3].critic_loss['Main']


        # Update loss info saved fill at the end of each epoch
        np.save(track_path +'LossTrackers.npy', ModelTrackers)

        # Save actions from the current epoch
        np.save(track_path +'ActionTrackers.npy', self.ActionTrackers)

        self.ModelTrackers = ModelTrackers


    def constructModelTrackers(self):

        ModelTrackers = {
            "salary_Eqty_arr"               :   np.array([]),
            "adm_Eqty_arr"                  :   np.array([]),    
            "adm_Rts_arr"                   :   np.array([]),  
            "grad_undergrad_Eqty_arr"       :   np.array([]),
            "grad_masters_Eqty_arr"         :   np.array([]), 
            "grad_doctorate_Eqty_arr"       :   np.array([]),
            "grad_undergrad_Rt_arr"         :   np.array([]),
            "grad_masters_Rt_arr"           :   np.array([]),
            "grad_doctorate_Rt_arr"         :   np.array([]),
            "emp_profits_arr"               :   np.array([]),
            "policy_univ_adm_loss_arr"      :   np.array([]),
            "policy_univ_budget_loss_arr"   :   np.array([]),
            "policy_emp_loss_arr"           :   np.array([]),
            "policy_planner_loss_arr"       :   np.array([]),
            "value_univ_adm_loss_arr"       :   np.array([]),
            "value_univ_budget_loss_arr"    :   np.array([]),
            "value_emp_loss_arr"            :   np.array([]),
            "value_planner_loss_arr"        :   np.array([])
            }

        return ModelTrackers


    def constructActionTrackers(self, action_szs):

        ActionTrackers = {}

        if action_szs[0] is not None:
            ActionTrackers["univ_adm_actions"] = np.zeros((0, action_szs[0]))

        if action_szs[1] is not None:
            ActionTrackers["univ_budget_actions"] = np.zeros((0, action_szs[1]))

        if action_szs[2] is not None:
            ActionTrackers["emp_actions"] = np.zeros((0, action_szs[2]))

        if action_szs[3] is not None:
            ActionTrackers["planner_actions"] = np.zeros((0, action_szs[3]))

        return ActionTrackers

        
    def step(self, observations):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        steps = []
        agent_i = 0
        for a in self.agents:
            if a is None:
                steps.append(None)
            else:
                steps.append(a.step(observations[agent_i]))
            agent_i+=1
        return steps
    
    def get_fake_actions(self, torch_agent_true_action):

        ''' Fake actions only are required when the obs is empty. Otherwise,
        torch_agent_true_actions = torch_agent_fake_actions'''
        torch_agent_true_action = torch_agent_true_action.detach().numpy()

        if torch_agent_true_action is not None:
            MM,NN = torch_agent_true_action.shape
            
            if MM == 0:
                act_4_train = np.zeros((1,NN)) # Action of all zeros if no observations provided
            else:
                act_4_train = torch_agent_true_action
            
            torch_agent_fake_action = Variable(torch.Tensor(act_4_train),requires_grad=False)

        else:
            torch_agent_fake_action = torch_agent_true_action
            
        return torch_agent_fake_action


    def update(self, sample, which_agents, batch_size, Q_plt_name, Action_plt_name, plt_num, clip_val=0.2, normalize_ad=True):        #
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
        """

        obs, acs, Gs, log_probs, rews, empty_obs, dones = sample

        # Loop through each agent
        for agent_i in range(len(self.agents)):
            curr_agent = self.agents[agent_i]
            
            if curr_agent is not None:
                # Convert Rewards to Torch
                Gs_torch = torch.Tensor([Gs[agent_i][i] for i in range(batch_size) if not empty_obs[agent_i][i]])
                log_probs_torch = torch.Tensor([[log_probs[agent_i][i]] for i in range(batch_size) if not empty_obs[agent_i][i]])
                acs_torch = torch.cat( tuple( acs[agent_i][i] for i in range(batch_size) if not empty_obs[agent_i][i] ) )

                # Compute advantages
                state_vals = torch.cat( tuple(curr_agent.critic(obs[0][i],obs[1][i],obs[2][i],obs[3][i]).reshape(1,1) for i in range(batch_size) if not empty_obs[agent_i][i])).detach()  # Intervene here

                A_k = Gs_torch - state_vals
                if normalize_ad:
                    A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # Normalize advantages

                # Update value net
                V = torch.cat( tuple(curr_agent.critic(obs[0][i], obs[1][i], obs[2][i], obs[3][i]).reshape(1,1) for i in range(batch_size) if not empty_obs[agent_i][i])) 
                critic_loss = MSELoss(V, Gs_torch)
                curr_agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
                curr_agent.critic_optimizer.step() 
                curr_agent.critic_loss['Main'] = np.append(curr_agent.critic_loss['Main'], critic_loss.detach().numpy())

                # Critic loss plots
                plt.plot(V.detach().numpy())                
                plt.plot(Gs_torch.detach().numpy())              
                plt.savefig(Q_plt_name+'agent_' + str(agent_i)+ '/plt_num_'+str(plt_num)+'.png')                                     
                plt.clf()                                                   
            
                # Get policy loss                                                                                                           
                curr_agent.policy_optimizer.zero_grad()                                                                               

                curr_outputs = [ curr_agent.policy(obs[agent_i][i])[2:4] for i in range(batch_size) if not empty_obs[agent_i][i] ]
                curr_log_probs = torch.cat([curr_outputs[i][0].unsqueeze(0) for i in range(len(curr_outputs)) ])
                curr_entropy = torch.cat([curr_outputs[i][1].unsqueeze(0) for i in range(len(curr_outputs)) ])

                # Action plots
                plt.hist(acs_torch.detach().numpy(), 50, range=[0, 1])                                                                     
                plt.xlabel('Actions')													
                plt.legend(['Group 1', 'Group 2'])												
                plt.savefig(Action_plt_name+'agent_' + str(agent_i)+ '/plt_num_'+str(plt_num)+'.png')                                                                                           
                plt.clf() 														   

                # Calculate ratios r(theta)
                ratios = torch.exp(curr_log_probs.unsqueeze(1) - log_probs_torch)
           
                # Calculate two surrogate loss terms in cliped loss
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1-clip_val, 1+clip_val) * A_k

                # Caluculate entropy       
                pol_loss = (-torch.min(surr1, surr2)-0.01*curr_entropy.unsqueeze(1)).mean() # Need negative sign to run Gradient Ascent 

                grad_norm = torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)                                        
                curr_agent.policy_optimizer.step()                                                                                                                           
                curr_agent.policy_loss = np.hstack((curr_agent.policy_loss, pol_loss.detach().numpy())) # save loss value for policy       
            

    def prep_training(self, device='cpu'):
        for a in self.agents:
            if a is not None:
                a.policy.train()
                a.critic.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                if a is not None:
                    a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                if a is not None:
                    a.critic = fn(a.critic)
            self.critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            if a is not None:
                a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                if a is not None:
                    a.policy = fn(a.policy)
            self.pol_dev = device


class MADDPG(object): # And DDPG
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task

    Credit: obtained from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/algorithms/maddpg.py

    Adapted by Author A
    """
    def __init__(self,
                 env,
                 which_agents,
                 policyTypes,
                 lmds=np.array([1,1,1]),
                 gamma=0.95,
                 tau=0.01,
                 lr_critic=0.01,
                 lr_policy=0.001,
                 hidden_dim=64,
                 Fair_scheme=0,
                 buffer_length=int(125*800),
                 track_path=None):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
        """

        self.which_agents = which_agents
        self.nagents = len(which_agents)
        self.agents = []
        self.track_path     = track_path
        self.alg_types= ['MADDPG']*3
        self.policyTypes = policyTypes
        self.lmds = np.expand_dims(np.array(lmds),1)
        self.gamma = gamma
        self.tau = tau
        self.lr_policy = lr_policy
        self.lr_critic = lr_critic
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        self.n_episodes = 0
        self.buffer_length = buffer_length
        self.N_steps = int(buffer_length*10)
        action_sizes        = [None, 6, None, 12]
        self.ActionTrackers = self.constructActionTrackers(action_sizes)
        self.ModelTrackers  = self.constructModelTrackers()

        
        # Create agents

        _,num_in_pol1 = env.world.agents[0].state.shape
        num_in_pol1 = num_in_pol1 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION
        input_dim_critic1 = num_in_pol1 + 1
                
        _,num_in_pol2 = env.world.agents[1].state.shape
        num_in_pol2 = num_in_pol2 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION
        input_dim_critic2 = num_in_pol2 + 6
            
        _,num_in_pol3 = env.world.agents[2].state.shape
        num_in_pol3 = num_in_pol3 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION
        input_dim_critic3 = num_in_pol3 + 1

        _,num_in_pol4 = env.world.agents[3].state.shape
        num_in_pol4 = num_in_pol4 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION
        input_dim_critic4 = num_in_pol4 + 12

        
        # Admissions
        if which_agents[0]:
            self.agents.append( AdmissionsAgent_MADDPG(num_in_pol1, policyTypes[0],
                                              input_dim_critic1, input_dim_critic2, input_dim_critic3, input_dim_critic4,
                                              hidden_dim, which_agents, lr_critic=lr_critic, lr_policy=lr_policy,
                                               Fair_scheme = Fair_scheme, N_steps=self.N_steps, decay_thresh=buffer_length) )
        else:
            self.agents.append(None)

        # Budget
        if which_agents[1]:
            self.agents.append( BudgetAgent_MADDPG(num_in_pol2, policyTypes[1],
                                             input_dim_critic1, input_dim_critic2, input_dim_critic3, input_dim_critic4,
                                             hidden_dim, which_agents, lr_critic=lr_critic, lr_policy=lr_policy,
                                              Fair_scheme = Fair_scheme, N_steps=self.N_steps, decay_thresh=buffer_length) )
        else:
            self.agents.append(None)

        # Employer
        if which_agents[2]:
            self.agents.append( EmployerAgent_MADDPG(num_in_pol3, policyTypes[2],
                                         input_dim_critic1, input_dim_critic2, input_dim_critic3, input_dim_critic4,
                                         hidden_dim, which_agents, lr_critic=lr_critic, lr_policy=lr_policy,
                                         Fair_scheme = Fair_scheme, N_steps=self.N_steps, decay_thresh=buffer_length) )
        else:
            self.agents.append(None)

        # PLanner
        if which_agents[3]:
            self.agents.append( PlannerAgent_MADDPG(num_in_pol4, policyTypes[3],
                                         input_dim_critic1, input_dim_critic2, input_dim_critic3, input_dim_critic4,
                                         hidden_dim, which_agents, lr_critic=lr_critic, lr_policy=lr_policy,
                                         Fair_scheme = Fair_scheme, N_steps=self.N_steps, decay_thresh=buffer_length) )
        else:
            self.agents.append(None)

    @property
    def policies(self):
        P = []
        for ii in range(len(self.agents)):
            if self.agents[ii] is not None:
                P.append(self.agents[ii].policy)
            else:
                P.append(None)
        return P

    @property
    def target_policies(self):
        T = []
        for ii in range(len(self.agents)):
            if self.agents[ii] is not None:
                T.append(self.agents[ii].target_policy)
            else:
                T.append(None)
        return T

    def reset_episode(self):

        self.reward_t = np.zeros((0,21))
        self.AgentRealization = None

    def update_rewards(self):

        track_path          = self.track_path
        ModelTrackers       = self.ModelTrackers

        # Calculate rewards from component functions
        Demographic_applied_univ        = self.reward_t[:,0:2]
        Demographic_admitted_univ       = self.reward_t[:,2:4]
        Demographic_went_to_undergrad   = self.reward_t[:,4:6]
        Demographic_graduated_undergrad = self.reward_t[:,6:8]
        Demographic_went_to_masters     = self.reward_t[:,8:10]
        Demographic_graduated_masters   = self.reward_t[:,10:12]
        Demographic_went_to_doctorate   = self.reward_t[:,12:14]
        Demographic_graduated_doctorate = self.reward_t[:,14:16]
        emp_profits                     = self.reward_t[:,16]
        Demographic_in_workforce        = self.reward_t[:,17:19]
        Demographic_salary              = self.reward_t[:,19:21]
            
        eps=1e-10

        # Salary Equity
        ave_salaries        = np.sum(Demographic_salary,axis=0)/np.sum(Demographic_in_workforce,axis=0)
        salary_Eqty         = -np.abs(ave_salaries[0]-ave_salaries[1])/(ave_salaries[0]+ave_salaries[1]+eps)

        # Admissions Equity
        adm_Rts             = np.sum(Demographic_admitted_univ,axis=0)/(np.sum(Demographic_applied_univ,axis=0)+eps)
        adm_Eqty            = -np.abs(adm_Rts[0]-adm_Rts[1])

        # Graduation Rate Equity (Undergrad)
        grad_undergrad      = np.sum(Demographic_graduated_undergrad[5:,:], axis=0)/(np.sum(Demographic_went_to_undergrad[2:-3,:], axis=0) + eps)
        grad_undergrad_Eqty = -np.abs(grad_undergrad[0] - grad_undergrad[1])

        # Graduation Rate Equity (Masters)
        grad_masters        = np.sum(Demographic_graduated_masters[7:,:], axis=0)/(np.sum(Demographic_went_to_masters[6:-1,:], axis=0)+eps)
        grad_masters_Eqty   = -np.abs(grad_masters[0] - grad_masters[1])

        # Graduation Rate Equity (Ph.D)
        grad_doctorate      = np.sum(Demographic_graduated_doctorate[12:,:], axis=0)/(np.sum(Demographic_went_to_doctorate[8:-4,:], axis=0)+eps)
        grad_doctorate_Eqty = -np.abs(grad_doctorate[0] - grad_doctorate[1])

        # Total Graduation Rates
        grad_undergrad_Rt   = np.sum(Demographic_graduated_undergrad[5:,:])/(np.sum(Demographic_went_to_undergrad[2:-3,:])+eps)
        grad_masters_Rt     = np.sum(Demographic_graduated_masters[7:,:])/(np.sum(Demographic_went_to_masters[6:-1,:])+eps)
        grad_doctorate_Rt   = np.sum(Demographic_graduated_doctorate[12:,:])/(np.sum(Demographic_went_to_doctorate[8:-4,:])+eps)

        # Total Admissions Rates
        adm_Rts             = np.sum(Demographic_admitted_univ)/(np.sum(Demographic_applied_univ)+eps)

        # Employer Profits
        emp_profits = np.sum(emp_profits)

        # Store model performance information for constructing loss values
        ModelTrackers["salary_Eqty_arr"]            = np.append(ModelTrackers["salary_Eqty_arr"], salary_Eqty)
        ModelTrackers["adm_Eqty_arr"]               = np.append(ModelTrackers["adm_Eqty_arr"], adm_Eqty)
        ModelTrackers["adm_Rts_arr"]                = np.append(ModelTrackers["adm_Rts_arr"], adm_Rts)
        ModelTrackers["grad_undergrad_Eqty_arr"]    = np.append(ModelTrackers["grad_undergrad_Eqty_arr"], grad_undergrad_Eqty)
        ModelTrackers["grad_masters_Eqty_arr"]      = np.append(ModelTrackers["grad_masters_Eqty_arr"], grad_masters_Eqty)
        ModelTrackers["grad_doctorate_Eqty_arr"]    = np.append(ModelTrackers["grad_doctorate_Eqty_arr"], grad_doctorate_Eqty)
        ModelTrackers["grad_undergrad_Rt_arr"]      = np.append(ModelTrackers["grad_undergrad_Rt_arr"], grad_undergrad_Rt)
        ModelTrackers["grad_masters_Rt_arr"]        = np.append(ModelTrackers["grad_masters_Rt_arr"], grad_masters_Rt)
        ModelTrackers["grad_doctorate_Rt_arr"]      = np.append(ModelTrackers["grad_doctorate_Rt_arr"], grad_doctorate_Rt)
        ModelTrackers["emp_profits_arr"]            = np.append(ModelTrackers["emp_profits_arr"], emp_profits)
        ModelTrackers["policy_univ_adm_loss_arr"]   = self.agents[0].policy_loss
        ModelTrackers["policy_univ_budget_loss_arr"]= self.agents[1].policy_loss
        ModelTrackers["policy_emp_loss_arr"]        = self.agents[2].policy_loss
        ModelTrackers["policy_planner_loss_arr"]    = self.agents[3].policy_loss
        ModelTrackers["value_univ_adm_loss_arr"]    = self.agents[0].critic_loss['Main']
        ModelTrackers["value_univ_budget_loss_arr"] = self.agents[1].critic_loss['Main']
        ModelTrackers["value_emp_loss_arr"]         = self.agents[2].critic_loss['Main']
        ModelTrackers["value_emp_loss_arr"]         = self.agents[3].critic_loss['Main']


        # Update loss info saved fill at the end of each epoch
        np.save(track_path +'LossTrackers.npy', ModelTrackers)

        # Save actions from the current epoch
        np.save(track_path +'ActionTrackers.npy', self.ActionTrackers)

        self.ModelTrackers = ModelTrackers


    def constructModelTrackers(self):

        ModelTrackers = {
            "salary_Eqty_arr"               :   np.array([]),
            "adm_Eqty_arr"                  :   np.array([]),    
            "adm_Rts_arr"                   :   np.array([]),  
            "grad_undergrad_Eqty_arr"       :   np.array([]),
            "grad_masters_Eqty_arr"         :   np.array([]), 
            "grad_doctorate_Eqty_arr"       :   np.array([]),
            "grad_undergrad_Rt_arr"         :   np.array([]),
            "grad_masters_Rt_arr"           :   np.array([]),
            "grad_doctorate_Rt_arr"         :   np.array([]),
            "emp_profits_arr"               :   np.array([]),
            "policy_univ_adm_loss_arr"      :   np.array([]),
            "policy_univ_budget_loss_arr"   :   np.array([]),
            "policy_emp_loss_arr"           :   np.array([]),
            "policy_planner_loss_arr"       :   np.array([]),
            "value_univ_adm_loss_arr"       :   np.array([]),
            "value_univ_budget_loss_arr"    :   np.array([]),
            "value_emp_loss_arr"            :   np.array([]),
            "value_planner_loss_arr"        :   np.array([])
            }

        return ModelTrackers


    def constructActionTrackers(self, action_szs):

        ActionTrackers = {}

        if action_szs[0] is not None:
            ActionTrackers["univ_adm_actions"] = np.zeros((0, action_szs[0]))

        if action_szs[1] is not None:
            ActionTrackers["univ_budget_actions"] = np.zeros((0, action_szs[1]))

        if action_szs[2] is not None:
            ActionTrackers["emp_actions"] = np.zeros((0, action_szs[2]))

        if action_szs[3] is not None:
            ActionTrackers["planner_actions"] = np.zeros((0, action_szs[3]))

        return ActionTrackers

        
    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            if a is not None:
                a.scale_noise(scale)
        
    def step(self, observations, ep_num, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        steps = []
        agent_i = 0
        for a in self.agents:
            if a is None:
                steps.append(None)
            else:
                steps.append(a.step(observations[agent_i], ep_num, explore=explore))
            agent_i+=1

        return steps

    
    def get_fake_actions(self, torch_agent_true_action):

        ''' Fake actions only are required when the obs is empty. Otherwise,
        torch_agent_true_actions = torch_agent_fake_actions'''
        torch_agent_true_action = torch_agent_true_action.detach().numpy()

        if torch_agent_true_action is not None:
            MM,NN = torch_agent_true_action.shape
            
            if MM == 0:
                act_4_train = np.zeros((1,NN)) # Action of all zeros if no observations provided
            else:
                act_4_train = torch_agent_true_action
            
            torch_agent_fake_action = Variable(torch.Tensor(act_4_train),requires_grad=False)

        else:
            torch_agent_fake_action = torch_agent_true_action
            
        return torch_agent_fake_action


    def component2rewards(self, rews, agent_i):

        eps     = 1e-10 
        comp_i  = rews[agent_i]
        rew_i   = []
        lmds    = self.lmds.flatten().astype('float')
        
        for ii in range(len(comp_i)):

            Demographic_applied_univ        = comp_i[ii][0:2]
            Demographic_admitted_univ       = comp_i[ii][2:4]
            Demographic_went_to_undergrad   = comp_i[ii][4:6]
            Demographic_graduated_undergrad = comp_i[ii][6:8]
            Demographic_went_to_masters     = comp_i[ii][8:10]
            Demographic_graduated_masters   = comp_i[ii][10:12]
            Demographic_went_to_doctorate   = comp_i[ii][12:14]
            Demographic_graduated_doctorate = comp_i[ii][14:16]
            emp_profits                     = comp_i[ii][16]
            Demographic_in_workforce        = comp_i[ii][17:19]
            Demographic_salary              = comp_i[ii][19:21]


            # Salary Equity
            ave_salaries        = Demographic_salary/(Demographic_in_workforce + eps)
            salary_Eqty         = -np.abs(ave_salaries[0]-ave_salaries[1])/(ave_salaries[0]+ave_salaries[1]+eps)

            # Admissions Equity
            adm_Rts             = Demographic_admitted_univ/(Demographic_applied_univ + eps)
            adm_Eqty            = -np.abs(adm_Rts[0]-adm_Rts[1])

            # Graduation Rate Equity (Undergrad)
            grad_undergrad      = Demographic_graduated_undergrad/(Demographic_went_to_undergrad + eps)
            grad_undergrad_Eqty = -np.abs(grad_undergrad[0] - grad_undergrad[1])

            # Graduation Rate Equity (Masters)
            grad_masters        = Demographic_graduated_masters/(Demographic_went_to_masters + eps)
            grad_masters_Eqty   = -np.abs(grad_masters[0] - grad_masters[1])

            # Graduation Rate Equity (Ph.D)
            grad_doctorate      = Demographic_graduated_doctorate/(Demographic_went_to_doctorate + eps)
            grad_doctorate_Eqty = -np.abs(grad_doctorate[0] - grad_doctorate[1])

            # Total Graduation Rates
            grad_undergrad_Rt   = np.sum(Demographic_graduated_undergrad)/np.sum(Demographic_went_to_undergrad + eps)
            grad_masters_Rt     = np.sum(Demographic_graduated_masters)/np.sum(Demographic_went_to_masters + eps)
            grad_doctorate_Rt   = np.sum(Demographic_graduated_doctorate)/np.sum(Demographic_went_to_doctorate + eps)

            # Total Admissions Rates
            adm_Rts             = np.sum(Demographic_admitted_univ)/(np.sum(Demographic_applied_univ)+eps)

            # Employer Profits
            emp_profits = emp_profits/6.0e5

            rew_i.append( lmds[0]*emp_profits + lmds[1]*adm_Rts + lmds[2]*grad_undergrad_Rt \
                          + lmds[3]*grad_masters_Rt + lmds[4]*grad_doctorate_Rt + lmds[5]*adm_Eqty \
                          + lmds[6]*grad_undergrad_Eqty + lmds[7]*grad_masters_Eqty + lmds[8]*grad_doctorate_Eqty \
                          + lmds[9]*salary_Eqty)
            
        return rew_i
    
    
    def update(self, sample, which_agents, batch_size, Q_plt_name, Action_plt_name, plt_num):        #
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """

        obs, acs, rews, next_obs_true, next_obs_fake, empty_obs, next_empty_obs, dones = sample
        for agent_i in range(len(which_agents)):

            if which_agents[agent_i] and sum(empty_obs[agent_i])/len(empty_obs[agent_i]): # Agent is not subsumed into the environment     #

                curr_agent = self.agents[agent_i]
                next_acs = []

    
                for pi, true_ob, fake_ob, empty_ob in zip(self.target_policies, next_obs_true, next_obs_fake, empty_obs):
                    if pi is not None:
                        curr_next_true_ac = [pi( true_ob[nn] ) for nn in range(batch_size) ]                                                           #                         
                        curr_next_fake_ac = [self.get_fake_actions( curr_next_true_ac[nn] ) for nn in range(batch_size) ]
                        next_acs.append([curr_next_fake_ac[nn] for nn in range(batch_size)])
                    else:
                        curr_next_true_ac = [None for nn in range(batch_size) ]
                        curr_next_fake_ac = [None for nn in range(batch_size) ]
                        next_acs.append([None for nn in range(batch_size)])

                        
                trgt_vf_in = self.augment_critic_Nobs(next_obs_fake, next_acs, agent_i, True)
    

                # Structure rewards from components
                rew_agent_i = self.component2rewards(rews, agent_i)
    
                # Get all target values
                target_value_main = torch.Tensor([ rew_agent_i[i] + (1-dones[agent_i][i]) * self.gamma *
                        curr_agent.target_critic(*trgt_vf_in[i]) for i in range(batch_size) if not empty_obs[agent_i][i]])                       # Intervene here

                    
                all_obs = [[ob[i] for i in range(batch_size)] for ob in obs]
                all_acs = [[ac[i] for i in range(batch_size)] for ac in acs]
                
                vf_in = self.augment_critic_Nobs(all_obs, all_acs, agent_i, True)
                actual_value_main = torch.cat( tuple(curr_agent.critic(*vf_in[i]).reshape(1) for i in range(batch_size) if not empty_obs[agent_i][i]))  # Intervene here
                
                # Main Critic Loss
                curr_agent.critic_optimizer.zero_grad()
                vf_loss = MSELoss(actual_value_main, target_value_main.detach())
                vf_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.05)
                curr_agent.critic_optimizer.step()
                curr_agent.critic_loss['Main'] = np.append(curr_agent.critic_loss['Main'], vf_loss.detach().numpy())

                # Critic loss plots for visualization
                plt.plot(actual_value_main.detach().numpy())                
                plt.plot(target_value_main.detach().numpy())                
                plt.savefig(Q_plt_name+'agent_' + str(agent_i)+ '/plt_num_'+str(plt_num)+'.png')  
                plt.clf()                                                   

                curr_agent.policy_optimizer.zero_grad()                                                                                    
                                                                                                                                       
                curr_pol_obs = [ obs[agent_i][i] for i in range(batch_size) ]                                                                        
                curr_pol_out = [ curr_agent.policy(curr_pol_obs[i]) for i in range(batch_size) ]                                             

                # Action plots for visualization														   					
                plt.hist(torch.cat(curr_pol_out).detach().numpy(), 50, range=[0, 1])                                                                      
                plt.xlabel('Actions')													
                plt.title('Histogram of Actions (ADM Thresholds) in Batch')									
                plt.legend(['Group 1', 'Group 2'])										    
                plt.savefig(Action_plt_name+'agent_' + str(agent_i)+ '/plt_num_'+str(plt_num)+'.png')                                                                                           
                plt.clf() 												   
                                                                                                                                            
                curr_ac_obs_vf_in = self.augment_critic_Nobs(curr_pol_obs, curr_pol_out, agent_i, False)
                
                vf_in = []                                                                                                               
                for i, pi, ob in zip(range(self.nagents), self.policies, obs):                                                          
                    if i == agent_i:                                                                                                   
                        vf_in.append(curr_ac_obs_vf_in)                                                                                   
                    elif ob[0] is None:                                                                                                
                        other_pol_obs = [ None for nn in range(batch_size) ]                                                              
                        other_pol_out = [ None for nn in range(batch_size) ]                                                             
                        other_ac_obs_vf_in = self.augment_critic_Nobs(other_pol_obs, other_pol_out, i, False)                          
                        vf_in.append(other_ac_obs_vf_in)                                                                               
                    else:                                                                                                                 
                        other_pol_obs = [ ob[nn] for nn in range(batch_size) ]                                                         
                        other_pol_out = [ pi(ob[nn]) for nn in range(batch_size) ]                                                     
                        other_ac_obs_vf_in = self.augment_critic_Nobs(other_pol_obs, other_pol_out, i, False)                                 
                                                                                                                                             
                        vf_in.append(other_ac_obs_vf_in)                                                                               
                                                                                                                                                                                                                                                                                         
                # Finalize structure                                                                                                       
                vf_in_final = [[vf_in[i][j] for i in range(len(vf_in))] for j in range(len(vf_in[0]))]                                     
                                                                                                                                           
                # Get policy loss                                                                                                              
                pol_loss = -torch.mean(torch.cat(tuple(curr_agent.critic(*vf_in_final[i]).reshape(1) for i in range(batch_size) )))                                                                                                              
                pol_loss.backward()                                                                                                        
                                                                                                                                           
                grad_norm = torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.05)                                            
                curr_agent.policy_optimizer.step()                                                                                                                                                                                                #
                curr_agent.policy_loss = np.hstack((curr_agent.policy_loss, pol_loss.detach().numpy())) # save loss value for policy       
                                                                                                                                            

    def augment_critic_Nobs(self, Nobs, Nactions, agent_i, combine_all = True):

        obsN = []

        # Individual vs combined actions
        if combine_all:
            N = len(Nobs[0])
        else:
            N = len(Nobs)

        for nn in range(N):

            # Individual vs combined actions
            if combine_all == True:
                obs_i = [Nobs[0][nn], Nobs[1][nn], Nobs[2][nn], Nobs[3][nn]]
                action_i = [Nactions[0][nn], Nactions[1][nn], Nactions[2][nn], Nactions[3][nn]]
            else:
                obs_i = Nobs[nn]
                action_i = Nactions[nn]

            obs_aug = self.augment_critic_obs(obs_i, action_i, agent_i, combine_all)
            obsN.append(obs_aug)

        return obsN            

    def augment_critic_obs(self, obs, actions, agent_i, combine_all = True):
        '''Depending on an agents actions, we need to construct the observation for the critic
        which is given by (obs_agent, action_agent) differently'''


        # Gather obs and actions
        if combine_all is True:
            obs_univ_adm, obs_univ_budget, obs_emp, obs_planner = obs
            actions_univ_adm, actions_univ_budget, actions_emp, actions_planner = actions
        else:
            if agent_i == 0:
                obs_univ_adm = obs
                actions_univ_adm = actions
            elif agent_i == 1:
                obs_univ_budget = obs
                actions_univ_budget = actions
            elif agent_i == 2:
                obs_emp = obs
                actions_emp = actions
            else:
                obs_planner = obs
                actions_planner = actions
                
        # Update admit agent
        if (combine_all is True) or (agent_i == 0):
    
            obs_ac_univ_adm = torch.cat((obs_univ_adm, actions_univ_adm), 1)

        if (combine_all is True) or (agent_i == 1):

            M,_ = obs_univ_budget.shape
            aug = Variable(torch.Tensor(np.ones((M,1))), requires_grad=False)
            aug_final = torch.matmul(aug,actions_univ_budget)
            obs_ac_univ_budget = torch.cat( (obs_univ_budget,aug_final), 1 )

        if (combine_all is True) or (agent_i == 2):

            obs_ac_emp = torch.cat((obs_emp, actions_emp), 1)

        if (combine_all is True) or (agent_i == 3):

            M,_ = obs_planner.shape
            aug = Variable(torch.Tensor(np.ones((M,1))), requires_grad=False)
            aug_final = torch.matmul(aug,actions_planner)
            obs_ac_planner = torch.cat( (obs_planner,aug_final), 1 )

        if combine_all is True:
            obs_ac = [obs_ac_univ_adm, obs_ac_univ_budget, obs_ac_emp, obs_ac_planner]
            
        else:
            if agent_i == 0:
                obs_ac = obs_ac_univ_adm
            elif agent_i == 1:
                obs_ac = obs_ac_univ_budget
            elif agent_i == 2:
                obs_ac = obs_ac_emp
            else:
                obs_ac = obs_ac_planner

        return obs_ac

    

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            if a is not None:
                soft_update(a.target_critic, a.critic, self.tau)
                soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='cpu'):
        for a in self.agents:
            if a is not None:
                a.policy.train()
                a.critic.train()
                a.target_policy.train()
                a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                if a is not None:
                    a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                if a is not None:
                    a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                if a is not None:
                    a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                if a is not None:
                    a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            if a is not None:
                a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                if a is not None:
                    a.policy = fn(a.policy)
            self.pol_dev = device



