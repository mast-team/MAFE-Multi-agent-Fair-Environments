import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents_fair_082024 import AdmissionAgent_MAAC, DisburseAgent_MAAC, DebtAgent_MAAC, AdmissionAgent_MADDPG, DisburseAgent_MADDPG, DebtAgent_MADDPG, AdmissionAgent_MAPPO, DisburseAgent_MAPPO, DebtAgent_MAPPO 
from utils.critics_fair_03272024 import LoanAttentionCritic
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt

MSELoss = torch.nn.MSELoss()


class CrossEntropyMethod(object):
    """
    Wrapper class for training cross entropy method
    """
    def __init__(self,
                 env,
                 policyTypes,
                 AwareOrNot             = True,
                 lmds                   = {"profits":1/6, "adm_Rts":1/6, "default_Rts":1/6, "adm_Eqty_Rts":1/6, "wait_Rts_Eqty":1/6, "default_Rts_Eqty":1/6},
                 eq_flags               = [False,False,False],
                 track_path             = None,
                 save_path              = [],
                 p_elite                = 0.2,
                 converge_flag          = True,
                 opt_type               = "independent",
                 N_epochs               = 50,
                 N_episodes             = 100,
                 episode_length         = 100,
                 adm_mu_scale           = 0,
                 adm_var_scale          = 10,
                 adm_mu_scale_b         = 0,
                 adm_var_scale_b        = 10,
                 disburse_mu_scale      = 0,
                 disburse_var_scale     = 1,
                 disburse_mu_scale_b    = 0,
                 disburse_var_scale_b   = 1,
                 debt_mu_scale          = 0,
                 debt_var_scale         = 10,
                 debt_mu_scale_b        = 0,
                 debt_var_scale_b       = 10,
                 aug_state_flag         = False): 

        # Layer information (adjacent values make up dimensions of a layer. Thus, N entries mean N-1 hidden layers (plus postential last output))
        intervene_Struct = []

        # Admissions Agent dimension info
        # Non-augmented features
        if aug_state_flag == False:
            _,num_in_pol1 = env.world.agents[0].state.shape
                
        # Augmented features
        else:
            _,num_in_pol1 = env.world.agents[0].aug_state.shape

        if policyTypes[0] == 1:
            if AwareOrNot and not eq_flags[0]: # Output different threshold for both groups, w/ sens attr feature
                layer_szs_admit  = [num_in_pol1,2]
            else:                               # No access to sensitive attribute so 1 threshold for everyone.
                layer_szs_admit  = [num_in_pol1,1]
            intervene_Struct.append(1)
            
        elif policyTypes[0] == 2:
            layer_szs_admit = [num_in_pol1,1]  # Output scores w/o sens attr feature
            intervene_Struct.append(2) 
        else:
            layer_szs_admit = None
            intervene_Struct.append(None)
                

        # Disbursement Agent dimension info
        # Non-augmented features
        if aug_state_flag == False:
            _,num_in_pol2 = env.world.agents[1].state.shape
            
        # Augmented features
        else:
            _,num_in_pol2 = env.world.agents[1].aug_state.shape

        # Disburse (Only score functionality)
        if policyTypes[1] == 1 or policyTypes[1] == 2:

            layer_szs_disburse  = [num_in_pol2,num_in_pol2+1] # Output scores w/ sens attr feature
            if policyTypes[1] == 1: 
                intervene_Struct.append(1)
            else:
                intervene_Struct.append(2)
            
        else:
            layer_szs_disburse = None
            intervene_Struct.append(None)
            

        # Debt Agent dimension info
        # Non-augmented features
        if aug_state_flag == False:
            _,num_in_pol3 = env.world.agents[2].state.shape
        else:
            _,num_in_pol3 = env.world.agents[2].aug_state.shape
                
        # Debt
        if policyTypes[2] == 1:
            layer_szs_debt  = [num_in_pol3,1]   # Output individual debt forgivenss w/o sens attr feature
            intervene_Struct.append(1)

        elif policyTypes[2] == 2:
            if AwareOrNot and not eq_flags[2]: # Output different forgivenss percentage for both groups, w/ sens attr feature
                layer_szs_debt  = [num_in_pol3,2]
            else:                               # No access to sensitive attribute so 1 forgivenss percentage for everyone.
                layer_szs_debt  = [num_in_pol3,1]
            intervene_Struct.append(2)
        else:
            layer_szs_debt = None
            intervene_Struct.append(None)
        
            
        self.AwareOrNot         = AwareOrNot
        layer_szs               =[layer_szs_admit, layer_szs_disburse, layer_szs_debt]
        self.agents             = None
        self.obs                = None
        self.actions            = [None, None, None]
        self.epoch              = 0
        self.layer_szs          = layer_szs
        self.p_elite            = p_elite
        self.converge_flag      = converge_flag
        self.eq_admit_flag      = eq_flags[0]
        self.eq_disburse_flag   = eq_flags[1]
        self.eq_debt_flag       = eq_flags[2]
        self.mean_elite_rewards = -np.inf
        self.intervene_Struct   = intervene_Struct
        self.lmds               = lmds
        self.policyTypes        = policyTypes
        self.opt_type           = opt_type
        self.LossTrackers       = self.constructLossTrackers(N_episodes)
        action_sizes            = [layer_szs_admit[-1], None, layer_szs_debt[-1]]
        self.ActionTrackers     = self.constructActionTrackers(action_sizes, N_epochs, N_episodes, episode_length-1)
        self.track_path         = track_path
        self.dist_agents        = self.construct_intervention_dictionary(layer_szs      =layer_szs,
                                                                    eq_admit_flag       =self.eq_admit_flag,
                                                                    eq_disburse_flag    =self.eq_disburse_flag,
                                                                    eq_debt_flag        =self.eq_debt_flag,
                                                                    save_path           =save_path,
                                                                    adm_mu_scale        =adm_mu_scale,
                                                                    adm_var_scale       =adm_var_scale,
                                                                    adm_mu_scale_b      =adm_mu_scale_b,
                                                                    adm_var_scale_b     =adm_var_scale_b,
                                                                    disburse_mu_scale   =disburse_mu_scale,
                                                                    disburse_var_scale  =disburse_var_scale,
                                                                    disburse_mu_scale_b =disburse_mu_scale_b,
                                                                    disburse_var_scale_b=disburse_var_scale_b,
                                                                    debt_mu_scale       =debt_mu_scale,
                                                                    debt_var_scale      =debt_var_scale,
                                                                    debt_mu_scale_b     =debt_mu_scale_b,
                                                                    debt_var_scale_b    =debt_var_scale_b)

    def reset_epoch(self):

        # Set up trackers for learning models
        self.ModelTrackers = self.constructModelTrackers()
        self.DictRealizations_N = []

    def reset_episode(self):

        self.reward_t = np.zeros((0,15))
        self.full_reward_t = np.zeros((0,5))
        self.AgentRealization = None

    def update_rewards(self):

        ModelTrackers       = self.ModelTrackers

        eps=1e-10
        profits          = self.reward_t[:,0]
        applied_w        = self.reward_t[:,1]
        applied_nw       = self.reward_t[:,2]
        accepted_w       = self.reward_t[:,3]
        accepted_nw      = self.reward_t[:,4]
        wait_w           = self.reward_t[:,5]
        wait_nw          = self.reward_t[:,6]
        disbursed_w      = self.reward_t[:,7]
        disbursed_nw     = self.reward_t[:,8]
        default_w        = self.reward_t[:,9]
        default_nw       = self.reward_t[:,10]
        score_w          = self.reward_t[:,11]
        score_nw         = self.reward_t[:,12]
        terminated_w     = self.reward_t[:,13]
        terminated_nw    = self.reward_t[:,14]

        # Normalization
        ave_wait_w = np.sum(wait_w)/(np.sum(disbursed_w)+eps)
        ave_wait_nw = np.sum(wait_nw)/(np.sum(disbursed_nw)+eps)
        norm_wait = ave_wait_w + ave_wait_nw 
        
        profits_rewards                 = np.mean(profits,0)
        adm_Rts_rewards                 = np.sum(accepted_w + accepted_nw)/(np.sum(applied_w + applied_nw) + eps)
        default_Rts_rewards             = -np.sum(default_w + default_nw)/(np.sum(terminated_w + terminated_nw) + eps)
        adm_Rts_Eqty_rewards            = -np.abs(np.sum(accepted_w)/(np.sum(applied_w)+eps) - np.sum(accepted_nw)/(np.sum(applied_nw)+eps)  )
        wait_Rts_Eqty_rewards           = -np.abs((ave_wait_w  - ave_wait_nw)/norm_wait) 
        default_Rts_Eqty_Rts_rewards    = -np.abs(np.sum(default_w)/(np.sum(terminated_w)+eps)-np.sum(default_nw)/(np.sum(terminated_nw)+eps))

        ModelTrackers["profits_arr"]            = np.append(ModelTrackers["profits_arr"]            , profits_rewards)
        ModelTrackers["adm_Rts_arr"]            = np.append(ModelTrackers["adm_Rts_arr"]            , adm_Rts_rewards)
        ModelTrackers["default_Rts_arr"]        = np.append(ModelTrackers["default_Rts_arr"]        , default_Rts_rewards)
        ModelTrackers["adm_Rts_Eqty_arr"]       = np.append(ModelTrackers["adm_Rts_Eqty_arr"]       , adm_Rts_Eqty_rewards)
        ModelTrackers["wait_Rts_Eqty_arr"]      = np.append(ModelTrackers["wait_Rts_Eqty_arr"]      , wait_Rts_Eqty_rewards)
        ModelTrackers["default_Rts_Eqty_arr"]   = np.append(ModelTrackers["default_Rts_Eqty_arr"]   , default_Rts_Eqty_Rts_rewards)

        self.ModelTrackers = ModelTrackers

            

    def construct_intervention_dictionary(self, layer_szs, eq_admit_flag, eq_disburse_flag, eq_debt_flag,
                                          save_path = [], train=True, adm_mu_scale=0, adm_var_scale=10, adm_mu_scale_b=0, adm_var_scale_b=10,
                                          disburse_mu_scale=0, disburse_var_scale=10, disburse_mu_scale_b=0, disburse_var_scale_b=10,
                                          debt_mu_scale=0, debt_var_scale=10, debt_mu_scale_b=0, debt_var_scale_b=10):

        # Create distribution parameters used to sample weights of the model
        AgentNames = ['admit','disburse','debt']
        dist_agents = {'admit':None, 'disburse':None, 'debt':None}

        # Import pretrained weights
        if len(save_path) > 0:
            dist_agents = np.load(save_path, allow_pickle=True).item()
            return dist_agents

        # Get layer sizes for each model
        layer_szs_admit, layer_szs_disburse, layer_szs_debt = layer_szs

        # Initialize agent storage
        for ii in range(len(self.intervene_Struct)):
            if self.intervene_Struct[ii] is not None:
                dist_agents[AgentNames[ii]] = {}

        if layer_szs[0] is not None:
            N_layers_adm = len(layer_szs[0])
        if layer_szs[1] is not None:
            N_layers_disburse = len(layer_szs[1])
        if layer_szs[2] is not None:
            N_layers_debt  = len(layer_szs[2])

        # Create admissions weights
        if self.intervene_Struct[0] is not None:
            for ii in range(N_layers_adm-1):

                dist_agents['admit']['admit_mu'+str(ii)] = np.ones((layer_szs_admit[ii], layer_szs_admit[ii+1]))*adm_mu_scale   # Mean of weights dist
                dist_agents['admit']['admit_var'+str(ii)] = np.ones((layer_szs_admit[ii], layer_szs_admit[ii+1]))*adm_var_scale # Std of weights dist
                dist_agents['admit']['admit_mu_b'+str(ii)] = np.ones((1,layer_szs_admit[ii+1]))*adm_mu_scale_b                                       # Mean of bias dist
                dist_agents['admit']['admit_var_b'+str(ii)] = np.ones((1,layer_szs_admit[ii+1]))*adm_var_scale_b                                      # Std of bias dist

            # This final layer follows the averaging later
            if self.intervene_Struct[0] == 1 and self.AwareOrNot and not eq_admit_flag:
                dist_agents['admit']['admit_mu'+str(N_layers_adm-1)] = np.ones((1,2))*adm_mu_scale                                             # Mean of weights dist
                dist_agents['admit']['admit_var'+str(N_layers_adm-1)] = np.ones((1,2))*adm_var_scale                                           # Std of weights dist
                dist_agents['admit']['admit_mu_b'+str(N_layers_adm-1)] = np.ones((1,2))*adm_mu_scale_b                                           # Mean of bias dist
                dist_agents['admit']['admit_var_b'+str(N_layers_adm-1)] = np.ones((1,2))*adm_var_scale_b                                         # Std of bias dist

            else:
                dist_agents['admit']['admit_mu'+str(N_layers_adm-1)] = np.ones((1,1))*adm_mu_scale                                             # Mean of weights dist
                dist_agents['admit']['admit_var'+str(N_layers_adm-1)] = np.ones((1,1))*adm_var_scale                                           # Std of weights dist
                dist_agents['admit']['admit_mu_b'+str(N_layers_adm-1)] = np.ones((1,1))*adm_mu_scale_b                                           # Mean of bias dist
                dist_agents['admit']['admit_var_b'+str(N_layers_adm-1)] = np.ones((1,1))*adm_var_scale_b                                         # Std of bias dist

        # Create disbursement weights
        if self.intervene_Struct[1] is not None:
            for ii in range(N_layers_disburse):

                dist_agents['disburse']['disburse_mu'+str(ii)] = np.ones((layer_szs_disburse[ii], 1))*disburse_mu_scale   # Mean of weights dist
                dist_agents['disburse']['disburse_var'+str(ii)] = np.ones((layer_szs_disburse[ii],1))*disburse_var_scale  # Std of weights dist
                dist_agents['disburse']['disburse_mu_b'+str(ii)] = np.ones((1,1))*disburse_mu_scale_b                     # Mean of bias dist
                dist_agents['disburse']['disburse_var_b'+str(ii)] = np.ones((1,1))*disburse_var_scale_b                   # Std of bias dist

        
        # Create debt forgiveness weights
        if self.intervene_Struct[2] is not None:
            for ii in range(N_layers_debt-1):

                dist_agents['debt']['debt_mu'+str(ii)] = np.ones((layer_szs_debt[ii], layer_szs_debt[ii+1]))*debt_mu_scale    # Mean of weights dist
                dist_agents['debt']['debt_var'+str(ii)] = np.ones((layer_szs_debt[ii], layer_szs_debt[ii+1]))*debt_var_scale  # Std of weights dist
                dist_agents['debt']['debt_mu_b'+str(ii)] = np.ones((1,layer_szs_debt[ii+1]))*debt_mu_scale_b                                       # Mean of bias dist
                dist_agents['debt']['debt_var_b'+str(ii)] = np.ones((1,layer_szs_debt[ii+1]))*debt_var_scale_b                                     # Std of bias dist

            # This final layer follows the averaging later
            if self.intervene_Struct[2] == 2 and self.AwareOrNot and not eq_debt_flag:
                dist_agents['debt']['debt_mu'+str(N_layers_debt-1)] = np.ones((1,2))*debt_mu_scale                                             # Mean of weights dist
                dist_agents['debt']['debt_var'+str(N_layers_debt-1)] = np.ones((1,2))*debt_var_scale                                           # Std of weights dist
                dist_agents['debt']['debt_mu_b'+str(N_layers_debt-1)] = np.ones((1,2))*debt_mu_scale_b                                           # Mean of bias dist
                dist_agents['debt']['debt_var_b'+str(N_layers_debt-1)] = np.ones((1,2))*debt_var_scale_b                                         # Std of bias dist

            else:
                dist_agents['debt']['debt_mu'+str(N_layers_debt-1)] = np.ones((1,1))*debt_mu_scale                                             # Mean of weights dist
                dist_agents['debt']['debt_var'+str(N_layers_debt-1)] = np.ones((1,1))*debt_var_scale                                           # Std of weights dist
                dist_agents['debt']['debt_mu_b'+str(N_layers_debt-1)] = np.ones((1,1))*debt_mu_scale_b                                           # Mean of bias dist
                dist_agents['debt']['debt_var_b'+str(N_layers_debt-1)] = np.ones((1,1))*debt_var_scale_b                                         # Std of bias dist

        return dist_agents

    def sample_intervention_parameters(self):

        # Sample the weights of the current model.

        AgentNames = ['admit','disburse','debt']
        self.AgentRealization = {'admit':None, 'disburse':None, 'debt':None}
        for ii in range(3):

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

                CurrDict_store[AgentNames[ii]+"_w"+str(jj)] = Params_w
                CurrDict_store[AgentNames[ii]+"_b"+str(jj)] = Params_b

            self.AgentRealization[AgentNames[ii]] = CurrDict_store


    def apply_intervention_model(self, obs):

        # Given module that we want to get result for and the associated observation, we gather appropriate weights from Model_Dict for this module and apply then to the data

        # Get current module's weights
        if self.AgentRealization['admit'] is not None:
            AdmitDict = self.AgentRealization['admit']
            data_admit = obs[0]

        if self.AgentRealization['disburse'] is not None:
            DisburseDict = self.AgentRealization['disburse']
            data_disburse = obs[1]

        if self.AgentRealization['debt'] is not None:
            DebtDict = self.AgentRealization['debt']
            data_debt = obs[2]
            
        # Apply the weights appropriately

        # ADMIT
        if self.intervene_Struct[0] is not None:

            # Run data through beginning model layers
            N = int(len(AdmitDict)/2)        # Divide by 2 since bias and weights each added for one layer, doubling its size
            
            if self.intervene_Struct[0] == 1:

                for ii in range(N-1):
                    data_admit = self.sig( data_admit@AdmitDict["admit_w"+str(ii)] + AdmitDict["admit_b"+str(ii)] )

                # Average output and run through final layer
                data_admit = self.sig( np.mean(data_admit,axis=0)*AdmitDict["admit_w"+str(N-1)] + AdmitDict["admit_b"+str(N-1)] )

            elif self.intervene_Struct[0] == 2:

                for ii in range(N-1):
                    data_admit = data_admit@AdmitDict["admit_w"+str(ii)] + AdmitDict["admit_b"+str(ii)] 

                    if ii < N-2:
                        data_admit = self.sig(data_admit)

                    else:
                        # Apply Laplacian Matrix (without storing whole thing)
                        mm, nn = data_admit.shape
                        scores = np.zeros((mm,nn))
                        lap_row = -np.ones((mm,nn))*(1/(mm-1 + 0.00000001))
                        for m in range(mm):
                            lap_row_curr = lap_row.copy()
                            lap_row_curr[m] = 1
                            scores[m] = lap_row_curr.T@data_admit
                        data_admit = self.sig( scores*AdmitDict["admit_w"+str(N-1)] + AdmitDict["admit_b"+str(N-1)] )
                        
            else: # Invalid intervention type
                raise ValueError

            # To deal with no customer situation
            if data_admit.shape[0]>0:
                if np.isnan(data_admit[0,0]):
                    _,NN = data_admit.shape
                    data_admit = np.zeros((0,NN))
            self.actions[0] = data_admit
        else:
            self.actions[0] = None

        # DISBURSE
        if self.intervene_Struct[1] is not None:

            # Run data through beginning model layers
            N = int(len(DisburseDict)/2)        # Divide by 2 since bias and weights each added for one layer, doubling its size

            if self.intervene_Struct[1] == 1:
                
                # init params
                M_batch, N_feat = data_disburse.shape
                data_disburse_orig = data_disburse.copy()

                # Run data through beginning model layers
                N = int(len(DisburseDict)/2)        # Divide by 2 since bias and weights each added for one layer, doubling its size
                for ii in range(N-1):
                    data_disburse = self.sig( data_disburse@DisburseDict["disburse_w"+str(ii)] + DisburseDict["disburse_b"+str(ii)] )

                # Average output and run through final layer
                data_disburse = np.mean(data_disburse,axis=0)

                # Append global feature to data and run through final MLP layer
                data_disburse = np.hstack((data_disburse_orig, np.ones((M_batch,1))*data_disburse))
                ii += 1
                data_disburse = self.sig( data_disburse@DisburseDict["disburse_w"+str(ii)] + DisburseDict["disburse_b"+str(ii)] )

            elif self.intervene_Struct[1] == 2:
                for ii in range(N-1):
                    data_disburse = self.sig( data_disburse@DisburseDict["disburse_w"+str(ii)] + DisburseDict["disburse_b"+str(ii)] )

                # Average output and run through final layer
                data_disburse = self.sig( np.mean(data_disburse,axis=0)*DisburseDict["disburse_w"+str(N-1)] + DisburseDict["disburse_b"+str(N-1)] )

            else: # Invalid intervention type
                raise ValueError

            # To deal with no customer situation
            if data_disburse.shape[0]>0:
                if np.isnan(data_disburse[0,0]):
                    _,NN = data_disburse.shape
                    data_disburse = np.zeros((0,NN))
            self.actions[1] = data_disburse#.flatten()?

        else:
            self.actions[1] = None

        # DEBT
        if self.intervene_Struct[2] is not None:

            # Run data through beginning model layers
            N = int(len(DebtDict)/2)        # Divide by 2 since bias and weights each added for one layer, doubling its size
            
            if self.intervene_Struct[2] == 1: # Individual Debt forgiveness
                for ii in range(N-1):
                    data_debt = data_debt@DebtDict["debt_w"+str(ii)] + DebtDict["debt_b"+str(ii)] 

                mm, nn = data_debt.shape
                debt_forgive_percentage = np.zeros((mm,nn))
                lap_row = -np.ones((mm,nn))*(1/(mm-1 + 0.00000001))
                for m in range(mm):
                    lap_row_curr = lap_row.copy()
                    lap_row_curr[m] = 1
                    debt_forgive_percentage[m] = lap_row_curr.T@data_debt
                data_debt = self.sig( debt_forgive_percentage*DebtDict["debt_w"+str(N-1)] + DebtDict["debt_b"+str(N-1)] )

            elif self.intervene_Struct[2] == 2: # Global Debt forgiveness
                for ii in range(N-1):
                    data_debt = self.sig( data_debt@DebtDict["debt_w"+str(ii)] + DebtDict["debt_b"+str(ii)] )
                # Average output and run through final layer
                data_debt = self.sig( np.mean(data_debt,axis=0)*DebtDict["debt_w"+str(N-1)] + DebtDict["debt_b"+str(N-1)] )

            else: # Invalid intervention type (Add more intervention types later)
                raise ValueError

            # To deal with no customer situation
            if data_debt.shape[0]>0:
                if np.isnan(data_debt[:,0]):
                    _,NN = data_debt.shape
                    data_debt = np.zeros((0,NN))
            self.actions[2] = data_debt#.flatten()?

        else:
            self.actions[2] = None

        return self.actions

    # Sigmoid function
    def sig(self, x):
        return 1 / (1 + np.exp(-x))
   
    def step(self, weight=0.67):

        track_path                  = self.track_path
        DictRealizations_N          = self.DictRealizations_N
        adm_Rts_rewards             = self.ModelTrackers["adm_Rts_arr"]
        default_Rts_rewards         = self.ModelTrackers["default_Rts_arr"]
        adm_Rts_Eqty_rewards        = self.ModelTrackers["adm_Rts_Eqty_arr"]
        wait_Rts_Eqty_rewards       = self.ModelTrackers["wait_Rts_Eqty_arr"]
        default_Rts_Eqty_rewards    = self.ModelTrackers["default_Rts_Eqty_arr"]
        profits_rewards             = self.ModelTrackers["profits_arr"]
        LossTrackers                = self.LossTrackers
        opt_type                    = self.opt_type
        lmds                        = self.lmds
        layer_szs                   = self.layer_szs
        p_elite                     = self.p_elite 
        
        # Organize for looping

        # Training using cooperation or independent schemes
        lmd_profits            = lmds["profits"]
        lmd_adm_Rts            = lmds["adm_Rts"]
        lmd_default_Rts        = lmds["default_Rts"]
        lmd_adm_Eqty_Rts       = lmds["adm_Eqty_Rts"]
        lmd_wait_Rts_Eqty      = lmds["wait_Rts_Eqty"]
        lmd_default_Rts_Eqty   = lmds["default_Rts_Eqty"]

        eps=1e-10
        profits_rewards_norm            = (profits_rewards - np.mean(profits_rewards))/(np.std(profits_rewards)+eps)
        adm_Rts_rewards_norm            = (adm_Rts_rewards - np.mean(adm_Rts_rewards))/(np.std(adm_Rts_rewards)+eps)
        default_Rts_rewards_norm        = (default_Rts_rewards - np.mean(default_Rts_rewards))/(np.std(default_Rts_rewards)+eps)
        adm_Rts_Eqty_rewards_norm       = (adm_Rts_Eqty_rewards - np.mean(adm_Rts_Eqty_rewards))/(np.std(adm_Rts_Eqty_rewards)+eps)
        wait_Rts_Eqty_rewards_norm      = (wait_Rts_Eqty_rewards - np.mean(wait_Rts_Eqty_rewards))/(np.std(wait_Rts_Eqty_rewards)+eps)
        default_Rts_Eqty_rewards_norm   = (default_Rts_Eqty_rewards - np.mean(default_Rts_Eqty_rewards))/(np.std(default_Rts_Eqty_rewards)+eps)

        coop_rewards    = (lmd_profits*profits_rewards_norm                 + \
                            lmd_adm_Rts*adm_Rts_rewards_norm                + \
                            lmd_default_Rts*default_Rts_rewards_norm        + \
                            lmd_adm_Eqty_Rts*adm_Rts_Eqty_rewards_norm      + \
                            lmd_wait_Rts_Eqty*wait_Rts_Eqty_rewards_norm    + \
                            lmd_default_Rts_Eqty*default_Rts_Eqty_rewards_norm)/6

        bank_rewards    = (lmd_profits*profits_rewards_norm                 + \
                            lmd_adm_Rts*adm_Rts_rewards_norm                + \
                            lmd_default_Rts*default_Rts_rewards_norm        + \
                            lmd_adm_Eqty_Rts*adm_Rts_Eqty_rewards_norm      + \
                            lmd_wait_Rts_Eqty*wait_Rts_Eqty_rewards_norm    + \
                            lmd_default_Rts_Eqty*default_Rts_Eqty_rewards_norm)/6


        ind_rewards = [bank_rewards, bank_rewards, bank_rewards]
        if opt_type == 'independent':
            train_rewards   = ind_rewards.copy()
        else:
            train_rewards   = [coop_rewards, coop_rewards, coop_rewards]


        # Initialize
        names = ['admit', 'disburse', 'debt']
        N_layers = len(layer_szs[0])
        p_idx = int(p_elite*len(DictRealizations_N))    # Elite index: index at the p% mark

        for n_agent in range(3):

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
        names        = ["profits", "adm_Rts", "default_Rts", "adm_Rts_Eqty",
                        "wait_Rts_Eqty", "default_Rts_Eqty"]
        ind_rewards  = [profits_rewards, adm_Rts_rewards, default_Rts_rewards, adm_Rts_Eqty_rewards,
                         wait_Rts_Eqty_rewards, default_Rts_Eqty_rewards]

        for n_reward in range(len(names)):
            
            # Current reward
            curr_ind_rewards    = ind_rewards[n_reward]
            curr_name           = names[n_reward]

            # Store current disparity average rewards
            LossTrackers[curr_name+"_mean_rewards"]              = np.append(LossTrackers[curr_name + "_mean_rewards"], np.mean(curr_ind_rewards))
            LossTrackers[curr_name+"_rewards_arr"]               = np.vstack((LossTrackers[curr_name + "_rewards_arr"], curr_ind_rewards))    


        # Update loss info saved fill at the end of each epoch
        np.save(track_path +'LossTrackers.npy', LossTrackers)

        # Save actions from the current epoch
        np.save(track_path +'ActionTrackers.npy', self.ActionTrackers)
        
        # Save InterveneDict so we can continue training/ load later on
        np.save(track_path +"InterveneDict.npy", self.dist_agents)

        # Store loss info back in environment 
        self.LossTrackers = LossTrackers


    def constructLossTrackers(self, N_episodes):

        LossTrackers = {
            "adm_Rts_rewards_arr"                   :   np.zeros((0,N_episodes)),
            "default_Rts_rewards_arr"               :   np.zeros((0,N_episodes)),
            "adm_Rts_Eqty_rewards_arr"              :   np.zeros((0,N_episodes)),
            "wait_Rts_Eqty_rewards_arr"             :   np.zeros((0,N_episodes)),
            "default_Rts_Eqty_rewards_arr"          :   np.zeros((0,N_episodes)),
            "profits_rewards_arr"                   :   np.zeros((0,N_episodes)),

            "adm_Rts_mean_rewards"                  :   np.zeros((0)),
            "default_Rts_mean_rewards"              :   np.zeros((0)),
            "adm_Rts_Eqty_mean_rewards"             :   np.zeros((0)),
            "wait_Rts_Eqty_mean_rewards"            :   np.zeros((0)),
            "default_Rts_Eqty_mean_rewards"         :   np.zeros((0)),
            "profits_mean_rewards"                  :   np.zeros((0))

            }

        return LossTrackers


    def constructModelTrackers(self):

        ModelTrackers = {
            "adm_Rts_arr"           :   np.array([]),
            "default_Rts_arr"       :   np.array([]),    
            "adm_Rts_Eqty_arr"      :   np.array([]),  
            "wait_Rts_Eqty_arr"     :   np.array([]),
            "default_Rts_Eqty_arr"  :   np.array([]), 
            "profits_arr"           :   np.array([])
            }

        return ModelTrackers


    def constructActionTrackers(self, action_szs, N_epochs, N_episodes, episode_len):

        ActionTrackers = {}

        if action_szs[0] is not None:
            ActionTrackers["adm_actions"] = np.zeros((N_epochs, N_episodes, episode_len, action_szs[0]))

        if action_szs[1] is not None:
            ActionTrackers["disburse_actions"] = np.zeros((N_epochs, N_episodes, episode_len, action_szs[1]))

        if action_szs[2] is not None:
            ActionTrackers["debt_actions"] = np.zeros((N_epochs, N_episodes, episode_len, action_szs[2]))

        return ActionTrackers


class MAPPO(object): 
    """
    Wrapper class for PPO-esque (i.e. also MAPPO) agents in multi-agent task

    """

    def __init__(self,
                 env,
                 which_agents,
                 policyTypes,
                 AwareOrNot=True,
                 lmds=np.array([1,1,1]),
                 gamma=0.95,
                 lr_critic=0.01,
                 lr_policy=0.001,
                 hidden_dim=64,
                 eq_flags = [False, False, False],
                 Fair_scheme=0,
                 track_path = None,
                 aug_state_flag = False):
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

        self.which_agents   = which_agents
        self.nagents        = len(which_agents)
        self.eq_flags       = eq_flags
        self.agents         = []
        self.track_path     = track_path
        self.policyTypes    = policyTypes
        self.lmds           = np.expand_dims(np.array(lmds),1)
        self.gamma          = gamma
        self.lr_policy      = lr_policy
        self.lr_critic      = lr_critic
        self.pol_dev        = 'cpu'  # device for policies
        self.critic_dev     = 'cpu'  # device for critics
        self.niter          = 0
        self.Fair_scheme    = Fair_scheme
        self.aug_state_flag = aug_state_flag
        action_sizes        = [2, None, 2]
        self.ActionTrackers = self.constructActionTrackers(action_sizes)
        self.ModelTrackers  = self.constructModelTrackers()
        
        # Create agents

        # Admissions Agent dimension info
        if which_agents[0]:

            # Non-augmented features
            if aug_state_flag == False:
                _,num_in_pol1 = env.world.agents[0].state.shape
                
            # Augmented features
            else:
                _,num_in_pol1 = env.world.agents[0].aug_state.shape
            
            num_in_pol1 = num_in_pol1 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION
            num_in_critic1 = num_in_pol1 
                
        else:
            num_in_critic1 = None

        # Disbursement Agent dimension info
        if which_agents[1]:

            # Non-augmented features
            if aug_state_flag == False:
                _,num_in_pol2 = env.world.agents[1].state.shape
            
            # Augmented features
            else:
                _,num_in_pol2 = env.world.agents[1].aug_state.shape

            num_in_pol2 = num_in_pol2 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION
            num_in_critic2 = num_in_pol2 
            
        else:
            num_in_critic2 = None

        # Debt Agent dimension info
        if which_agents[2]:

            # Non-augmented features
            if aug_state_flag == False:
                _,num_in_pol3 = env.world.agents[2].state.shape
            else:
                _,num_in_pol3 = env.world.agents[2].aug_state.shape
            
            num_in_pol3 = num_in_pol3 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION
            num_in_critic3 = num_in_pol3 
                
        else:
            num_in_critic3 = None

        
        # Admissions
        if which_agents[0]:
            self.agents.append( AdmissionAgent_MAPPO(num_in_pol1, policyTypes[0], eq_flags[0], AwareOrNot,
                                              num_in_critic1, num_in_critic2, num_in_critic3,
                                              hidden_dim, which_agents, lr_critic=lr_critic, lr_policy=lr_policy) )
        else:
            self.agents.append(None)

        # Disbursement
        if which_agents[1]:
            self.agents.append( DisburseAgent_MAPPO(num_in_pol2, policyTypes[1], eq_flags[1], AwareOrNot,
                                             num_in_critic1, num_in_critic2, num_in_critic3,
                                             hidden_dim, which_agents, lr_critic=lr_critic, lr_policy=lr_policy) )
        else:
            self.agents.append(None)

        # Debt
        if which_agents[2]:
            self.agents.append( DebtAgent_MAPPO(num_in_pol3, policyTypes[2], eq_flags[2], AwareOrNot,
                                         num_in_critic1, num_in_critic2, num_in_critic3,
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

        self.reward_t = np.zeros((0,15))
        self.full_reward_t = np.zeros((0,5))
        self.AgentRealization = None

    def update_rewards(self):

        track_path          = self.track_path
        ModelTrackers       = self.ModelTrackers

        eps=1e-10
        profits          = self.reward_t[:,0]
        applied_w        = self.reward_t[:,1]
        applied_nw       = self.reward_t[:,2]
        accepted_w       = self.reward_t[:,3]
        accepted_nw      = self.reward_t[:,4]
        wait_w           = self.reward_t[:,5]
        wait_nw          = self.reward_t[:,6]
        disbursed_w      = self.reward_t[:,7]
        disbursed_nw     = self.reward_t[:,8]
        default_w        = self.reward_t[:,9]
        default_nw       = self.reward_t[:,10]
        score_w          = self.reward_t[:,11]
        score_nw         = self.reward_t[:,12]
        terminated_w     = self.reward_t[:,13]
        terminated_nw    = self.reward_t[:,14]

        # Normalization
        ave_wait_w = np.sum(wait_w)/(np.sum(disbursed_w)+eps)
        ave_wait_nw = np.sum(wait_nw)/(np.sum(disbursed_nw)+eps)
        norm_wait = ave_wait_w + ave_wait_nw 
        
        profits_rewards                 = np.mean(profits,0)
        adm_Rts_rewards                 = np.sum(accepted_w + accepted_nw)/(np.sum(applied_w + applied_nw) + eps)
        default_Rts_rewards             = -np.sum(default_w + default_nw)/(np.sum(terminated_w + terminated_nw) + eps)
        adm_Rts_Eqty_rewards            = -np.abs(np.sum(accepted_w)/(np.sum(applied_w)+eps) - np.sum(accepted_nw)/(np.sum(applied_nw)+eps)  )
        wait_Rts_Eqty_rewards           = -np.abs((ave_wait_w  - ave_wait_nw)/norm_wait) 
        default_Rts_Eqty_Rts_rewards    = -np.abs(np.sum(default_w)/(np.sum(terminated_w)+eps)-np.sum(default_nw)/(np.sum(terminated_nw)+eps))

        ModelTrackers["profits_arr"]                = np.append(ModelTrackers["profits_arr"]            , profits_rewards)
        ModelTrackers["adm_Rts_arr"]                = np.append(ModelTrackers["adm_Rts_arr"]            , adm_Rts_rewards)
        ModelTrackers["default_Rts_arr"]            = np.append(ModelTrackers["default_Rts_arr"]        , default_Rts_rewards)
        ModelTrackers["adm_Rts_Eqty_arr"]           = np.append(ModelTrackers["adm_Rts_Eqty_arr"]       , adm_Rts_Eqty_rewards)
        ModelTrackers["wait_Rts_Eqty_arr"]          = np.append(ModelTrackers["wait_Rts_Eqty_arr"]      , wait_Rts_Eqty_rewards)
        ModelTrackers["default_Rts_Eqty_arr"]       = np.append(ModelTrackers["default_Rts_Eqty_arr"]   , default_Rts_Eqty_Rts_rewards)
        ModelTrackers["policy_adm_loss_arr"]        = self.agents[0].policy_loss
        ModelTrackers["policy_disburse_loss_arr"]   = self.agents[1].policy_loss
        ModelTrackers["policy_debt_loss_arr"]       = self.agents[2].policy_loss
        ModelTrackers["value_adm_loss_arr"]         = self.agents[0].critic_loss['Main']
        ModelTrackers["value_disburse_loss_arr"]    = self.agents[1].critic_loss['Main']
        ModelTrackers["value_debt_loss_arr"]        = self.agents[2].critic_loss['Main']



        # Update loss info saved fill at the end of each epoch
        np.save(track_path +'LossTrackers.npy', ModelTrackers)

        # Save actions from the current epoch
        np.save(track_path +'ActionTrackers.npy', self.ActionTrackers)

        self.ModelTrackers = ModelTrackers


    def constructModelTrackers(self):

        ModelTrackers = {
            "adm_Rts_arr"               :   np.array([]),
            "default_Rts_arr"           :   np.array([]),    
            "adm_Rts_Eqty_arr"          :   np.array([]),  
            "wait_Rts_Eqty_arr"         :   np.array([]),
            "default_Rts_Eqty_arr"      :   np.array([]), 
            "profits_arr"               :   np.array([]),
            "policy_adm_loss_arr"       :   np.array([]),
            "policy_disburse_loss_arr"  :   np.array([]),
            "policy_debt_loss_arr"      :   np.array([]),
            "value_adm_loss_arr"        :   np.array([]),
            "value_disburse_loss_arr"   :   np.array([]),
            "value_debt_loss_arr"       :   np.array([])
            }

        return ModelTrackers


    def constructActionTrackers(self, action_szs):

        ActionTrackers = {}

        if action_szs[0] is not None:
            ActionTrackers["adm_actions"] = np.zeros((0, action_szs[0]))

        if action_szs[1] is not None:
            ActionTrackers["disburse_actions"] = np.zeros((0, action_szs[1]))

        if action_szs[2] is not None:
            ActionTrackers["debt_actions"] = np.zeros((0, action_szs[2]))

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
                state_vals = torch.cat( tuple(curr_agent.critic(obs[0][i],obs[1][i],obs[2][i]).reshape(1,1) for i in range(batch_size) if not empty_obs[agent_i][i])).detach()  # Intervene here

                A_k = Gs_torch - state_vals
                if normalize_ad:
                    A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # Normalize advantages

                # Update value net
                V = torch.cat( tuple(curr_agent.critic(obs[0][i], obs[1][i], obs[2][i]).reshape(1,1) for i in range(batch_size) if not empty_obs[agent_i][i])) 
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
                plt.title('Histogram of Actions (ADM Thresholds) in Batch')									
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

    """
    def __init__(self,
                 env,
                 which_agents,
                 policyTypes,
                 AwareOrNot=True,
                 lmds=np.array([1,1,1]),
                 gamma=0.95,
                 tau=0.01,
                 lr_critic=0.01,
                 lr_policy=0.001,
                 hidden_dim=64,
                 eq_flags = [False, False, False],
                 Fair_scheme=0,
                 track_path = None,
                 aug_state_flag = False,
                 buffer_length=int(125*800)):
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

        self.which_agents   = which_agents
        self.nagents        = len(which_agents)
        self.eq_flags       = eq_flags
        self.agents         = []
        self.track_path     = track_path
        self.alg_types      = ['MADDPG']*3
        self.policyTypes    = policyTypes
        self.lmds           = np.expand_dims(np.array(lmds),1)
        self.gamma          = gamma
        self.tau            = tau
        self.lr_policy      = lr_policy
        self.lr_critic      = lr_critic
        self.pol_dev        = 'cpu'  # device for policies
        self.critic_dev     = 'cpu'  # device for critics
        self.trgt_pol_dev   = 'cpu'  # device for target policies
        self.trgt_critic_dev= 'cpu'  # device for target critics
        self.niter          = 0
        self.n_episodes     = 0
        self.aug_state_flag = aug_state_flag
        self.buffer_length  = buffer_length
        self.N_steps        = int(buffer_length*10)
        action_sizes        = [2, None, 2]
        self.ActionTrackers = self.constructActionTrackers(action_sizes)
        self.ModelTrackers  = self.constructModelTrackers()
        
        # Create agents

        # Admissions Agent dimension info
        if which_agents[0]:

            # Non-augmented features
            if aug_state_flag == False:
                _,num_in_pol1 = env.world.agents[0].state.shape
                
            # Augmented features
            else:
                _,num_in_pol1 = env.world.agents[0].aug_state.shape
            
            num_in_pol1 = num_in_pol1 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION

            if policyTypes[0] == 1 and not eq_flags[0]:  # Accounts for action as input to critic on top of the state info
                input_dim_critic1 = num_in_pol1 + 1*2
            else:
                input_dim_critic1 = num_in_pol1 + 1*1
                
        else:
            input_dim_critic1 = None

        # Disbursement Agent dimension info
        if which_agents[1]:

            # Non-augmented features
            if aug_state_flag == False:
                _,num_in_pol2 = env.world.agents[1].state.shape
            
            # Augmented features
            else:
                _,num_in_pol2 = env.world.agents[1].aug_state.shape

            num_in_pol2 = num_in_pol2 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION

            input_dim_critic2 = num_in_pol2 + 1*1
            
        else:
            input_dim_critic2 = None

        # Debt Agent dimension info
        if which_agents[2]:

            # Non-augmented features
            if aug_state_flag == False:
                _,num_in_pol3 = env.world.agents[2].state.shape
            else:
                _,num_in_pol3 = env.world.agents[2].aug_state.shape
            
            num_in_pol3 = num_in_pol3 + 1           # ACCOUNTS FOR EMPTY OBSERVATION EXTRA FLAG DIMENSION

            if policyTypes[2] == 2 and not eq_flags[2]:  # Accounts for action as input to critic on top of the state info
                input_dim_critic3 = num_in_pol3 + 1*2
            else:
                input_dim_critic3 = num_in_pol3+1*1
                
        else:
            input_dim_critic3 = None

        
        # Admissions
        if which_agents[0]:
            self.agents.append( AdmissionAgent_MADDPG(num_in_pol1, policyTypes[0], eq_flags[0], AwareOrNot,
                                              input_dim_critic1, input_dim_critic2, input_dim_critic3,
                                              hidden_dim, which_agents, lr_critic=lr_critic, lr_policy=lr_policy,
                                               Fair_scheme = Fair_scheme, N_steps=self.N_steps, decay_thresh=buffer_length) )
        else:
            self.agents.append(None)

        # Disbursement
        if which_agents[1]:
            self.agents.append( DisburseAgent_MADDPG(num_in_pol2, policyTypes[1], eq_flags[1], AwareOrNot,
                                             input_dim_critic1, input_dim_critic2, input_dim_critic3,
                                             hidden_dim, which_agents, lr_critic=lr_critic, lr_policy=lr_policy,
                                              Fair_scheme = Fair_scheme, N_steps=self.N_steps, decay_thresh=buffer_length) )
        else:
            self.agents.append(None)

        # Debt
        if which_agents[2]:
            self.agents.append( DebtAgent_MADDPG(num_in_pol3, policyTypes[2], eq_flags[2], AwareOrNot,
                                         input_dim_critic1, input_dim_critic2, input_dim_critic3,
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

        self.reward_t = np.zeros((0,15))
        self.full_reward_t = np.zeros((0,5))
        self.AgentRealization = None

    def update_rewards(self):

        track_path          = self.track_path
        ModelTrackers       = self.ModelTrackers

        eps=1e-10
        profits          = self.reward_t[:,0]
        applied_w        = self.reward_t[:,1]
        applied_nw       = self.reward_t[:,2]
        accepted_w       = self.reward_t[:,3]
        accepted_nw      = self.reward_t[:,4]
        wait_w           = self.reward_t[:,5]
        wait_nw          = self.reward_t[:,6]
        disbursed_w      = self.reward_t[:,7]
        disbursed_nw     = self.reward_t[:,8]
        default_w        = self.reward_t[:,9]
        default_nw       = self.reward_t[:,10]
        score_w          = self.reward_t[:,11]
        score_nw         = self.reward_t[:,12]
        terminated_w     = self.reward_t[:,13]
        terminated_nw    = self.reward_t[:,14]

        # Normalization
        ave_wait_w = np.sum(wait_w)/(np.sum(disbursed_w)+eps)
        ave_wait_nw = np.sum(wait_nw)/(np.sum(disbursed_nw)+eps)
        norm_wait = ave_wait_w + ave_wait_nw 
        
        profits_rewards                 = np.mean(profits,0)
        adm_Rts_rewards                 = np.sum(accepted_w + accepted_nw)/(np.sum(applied_w + applied_nw) + eps)
        default_Rts_rewards             = -np.sum(default_w + default_nw)/(np.sum(terminated_w + terminated_nw) + eps)
        adm_Rts_Eqty_rewards            = -np.abs(np.sum(accepted_w)/(np.sum(applied_w)+eps) - np.sum(accepted_nw)/(np.sum(applied_nw)+eps)  )
        wait_Rts_Eqty_rewards           = -np.abs((ave_wait_w  - ave_wait_nw)/norm_wait) 
        default_Rts_Eqty_Rts_rewards    = -np.abs(np.sum(default_w)/(np.sum(terminated_w)+eps)-np.sum(default_nw)/(np.sum(terminated_nw)+eps))

        ModelTrackers["profits_arr"]            = np.append(ModelTrackers["profits_arr"]            , profits_rewards)
        ModelTrackers["adm_Rts_arr"]            = np.append(ModelTrackers["adm_Rts_arr"]            , adm_Rts_rewards)
        ModelTrackers["default_Rts_arr"]        = np.append(ModelTrackers["default_Rts_arr"]        , default_Rts_rewards)
        ModelTrackers["adm_Rts_Eqty_arr"]       = np.append(ModelTrackers["adm_Rts_Eqty_arr"]       , adm_Rts_Eqty_rewards)
        ModelTrackers["wait_Rts_Eqty_arr"]      = np.append(ModelTrackers["wait_Rts_Eqty_arr"]      , wait_Rts_Eqty_rewards)
        ModelTrackers["default_Rts_Eqty_arr"]   = np.append(ModelTrackers["default_Rts_Eqty_arr"]   , default_Rts_Eqty_Rts_rewards)

        # Update loss info saved fill at the end of each epoch
        np.save(track_path +'LossTrackers.npy', ModelTrackers)

        # Save actions from the current epoch
        np.save(track_path +'ActionTrackers.npy', self.ActionTrackers)

        self.ModelTrackers = ModelTrackers


    def constructModelTrackers(self):

        ModelTrackers = {
            "adm_Rts_arr"           :   np.array([]),
            "default_Rts_arr"       :   np.array([]),    
            "adm_Rts_Eqty_arr"      :   np.array([]),  
            "wait_Rts_Eqty_arr"     :   np.array([]),
            "default_Rts_Eqty_arr"  :   np.array([]), 
            "profits_arr"           :   np.array([])
            }

        return ModelTrackers


    def constructActionTrackers(self, action_szs):

        ActionTrackers = {}

        if action_szs[0] is not None:
            ActionTrackers["adm_actions"] = np.zeros((0, action_szs[0]))

        if action_szs[1] is not None:
            ActionTrackers["disburse_actions"] = np.zeros((0, action_szs[1]))

        if action_szs[2] is not None:
            ActionTrackers["debt_actions"] = np.zeros((0, action_szs[2]))

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

            profits         = comp_i[ii][0,0]
            applied_w       = comp_i[ii][0,1]
            applied_nw      = comp_i[ii][0,2]
            accepted_w      = comp_i[ii][0,3]
            accepted_nw     = comp_i[ii][0,4]
            wait_w          = comp_i[ii][0,5]
            wait_nw         = comp_i[ii][0,6]
            disbursed_w     = comp_i[ii][0,7]
            disbursed_nw    = comp_i[ii][0,8]
            default_w       = comp_i[ii][0,9]
            default_nw      = comp_i[ii][0,10]
            score_w         = comp_i[ii][0,11]
            score_nw        = comp_i[ii][0,12]
            terminated_w    = comp_i[ii][0,13]
            terminated_nw   = comp_i[ii][0,14]

            # Normalization
            ave_wait_w = wait_w / ( disbursed_w + eps )
            ave_wait_nw = wait_nw / ( disbursed_nw +eps)
            norm_wait = ave_wait_w + ave_wait_nw
        
            profits_rewards                 = profits / 88753.49742194821
            adm_Rts_rewards                 = (accepted_w + accepted_nw)/( (applied_w + applied_nw) + eps)
            default_Rts_rewards             = -(default_w + default_nw)/( (terminated_w + terminated_nw) + eps)
            adm_Rts_Eqty_rewards            = -np.abs( accepted_w/(applied_w + eps) - accepted_nw/(applied_nw + eps)  )
            wait_Rts_Eqty_rewards           = -np.abs((ave_wait_w  - ave_wait_nw)/(norm_wait + eps) ) 
            default_Rts_Eqty_Rts_rewards    = -np.abs( default_w/(terminated_w+eps) - default_nw/(terminated_nw+eps) )

            rew_i.append( lmds[0]*profits_rewards + lmds[1]*adm_Rts_rewards + lmds[2]*default_Rts_rewards
                          + lmds[3]*adm_Rts_Eqty_rewards + lmds[4]*wait_Rts_Eqty_rewards + lmds[5]*default_Rts_Eqty_Rts_rewards)

        return rew_i
            
        
    
    def update(self, sample, which_agents, batch_size, Q_plt_name, Action_plt_name, plt_num):       
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

            if which_agents[agent_i]: # Agent is not subsumed into the environment  

                curr_agent = self.agents[agent_i]
                next_acs = []
    
                for pi, true_ob, fake_ob, empty_ob in zip(self.target_policies, next_obs_true, next_obs_fake, empty_obs):
                    if pi is not None:
                        curr_next_true_ac = [pi( true_ob[nn] ) for nn in range(batch_size) ]                                                                                   
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
                        curr_agent.target_critic(*trgt_vf_in[i]) for i in range(batch_size) if not empty_obs[agent_i][i]])                      

                all_obs = [[ob[i] for i in range(batch_size)] for ob in obs]
                all_acs = [[ac[i] for i in range(batch_size)] for ac in acs]

                vf_in = self.augment_critic_Nobs(all_obs, all_acs, agent_i, True)
                actual_value_main = torch.cat( tuple(curr_agent.critic(*vf_in[i]).reshape(1) for i in range(batch_size) if not empty_obs[agent_i][i]))  


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
                obs_i = [Nobs[0][nn], Nobs[1][nn], Nobs[2][nn]]
                action_i = [Nactions[0][nn], Nactions[1][nn], Nactions[2][nn]]
            else:
                obs_i = Nobs[nn]
                action_i = Nactions[nn]

            obs_aug = self.augment_critic_obs(obs_i, action_i, agent_i, combine_all)
            obsN.append(obs_aug)

        return obsN            

    def augment_critic_obs(self, obs, actions, agent_i, combine_all = True):
        '''Depending on an agents actions, we need to construct the observation for the critic
        which is given by (obs_agent, action_agent) differently'''

        # Operations must be tensor-differentiably-friendly

        policy_adm, policy_disburse, policy_debt = self.policyTypes

        # Gather obs and actions
        if combine_all is True:
            obs_adm, obs_disburse, obs_debt = obs
            actions_adm, actions_disburse, actions_debt = actions
        else:
            if agent_i == 0:
                obs_adm = obs
                actions_adm = actions
            elif agent_i == 1:
                obs_disburse = obs
                actions_disburse = actions
            else:
                obs_debt = obs
                actions_debt = actions
                
        # Update admit agent
        if (combine_all is True) or (agent_i == 0):
            if policy_adm == 1:
    
                # Convert to numpy and build structure to augment the matrix
                M,_ = obs_adm.shape
                aug = Variable(torch.Tensor(np.ones((M,1))), requires_grad=False)
                aug_final = torch.matmul(aug,actions_adm)
                obs_ac_adm = torch.cat( (obs_adm,aug_final), 1 )            
            
            elif policy_adm == 2:
                obs_ac_adm = torch.cat((obs_adm, actions_adm), 1)

            else: # Part of the environment
                obs_ac_adm = None

        if (combine_all is True) or (agent_i == 1):
            # Update disburse agent
            if policy_disburse == 1:
                obs_ac_disburse = torch.cat((obs_disburse, actions_disburse), 1)
                
            elif policy_disburse == 2:
                M,_ = obs_disburse.shape
                aug = Variable(torch.Tensor(np.ones((M,1))), requires_grad=False)
                aug_final = torch.matmul(aug,actions_disburse)
                obs_ac_disburse = torch.cat( (obs_disburse,aug_final), 1 )

            else:
                obs_ac_disburse = None

        if (combine_all is True) or (agent_i == 2):
            # Update debt agent
            if policy_debt == 1: # Individual intervention

                obs_ac_debt = torch.cat((obs_debt, actions_debt), 1)
            
            elif policy_debt == 2: # Global intervention

                # Convert to numpy and build structure to augment the matrix
                M,_ = obs_debt.shape
                aug = Variable(torch.Tensor(np.ones((M,1))), requires_grad=False)
                aug_final = torch.matmul(aug,actions_debt)
                obs_ac_debt = torch.cat( (obs_debt,aug_final), 1 )
                
            else:
                obs_ac_debt = None

        if combine_all is True:
            obs_ac = [obs_ac_adm, obs_ac_disburse, obs_ac_debt]
        else:
            if agent_i == 0:
                obs_ac = obs_ac_adm
            elif agent_i == 1:
                obs_ac = obs_ac_disburse
            else:
                obs_ac = obs_ac_debt

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



