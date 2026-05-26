import numpy as np

class CrossEntropyMethod(object):
    """
    Wrapper class for training cross entropy method
    """
    def __init__(self,
                 env,
                 policyTypes,
                 lmds               = {"ins_profits":0.142857,
                                       "mortRts":0.142857,
                                       "insRts":0.142857,
                                       "incidenceRts":0.142857,
                                       "mortRts_geoEqty":0.142857,
                                       "insRts_geoEqty":0.142857,
                                       "incidenceRts_geoEqty":0.142857},
                 save_path          = [],
                 p_elite            = 0.2,
                 tracking_path      = None,
                 opt_type           = "independent",
                 which_agents       = [True, True, True],
                 N_epochs           = 50,
                 N_episodes         = 100,
                 episode_length     = 100,
                 ins_mu_scale       = 0,
                 ins_var_scale      = 2,
                 ins_mu_scale_b     = 0,
                 ins_var_scale_b    = 2,
                 hosp_mu_scale      = 0,
                 hosp_var_scale     = 2,
                 hosp_mu_scale_b    = 0,
                 hosp_var_scale_b   = 2,
                 planner_mu_scale   = 0,
                 planner_var_scale  = 2,
                 planner_mu_scale_b = 0,
                 planner_var_scale_b= 2): 

        # Layer information (adjacent values make up dimensions of a layer. Thus, N entries mean N-1 hidden layers (plus postential last output))
        N_feat = {"ins":env.world.agents[0].state.shape[1], "hosp":env.world.agents[1].state.shape[1], "planner":env.world.agents[2].state.shape[1]}
        intervene_Struct = []

        # Insurance agent policy type
        if policyTypes[0] == 1:
            layer_szs_ins   = [N_feat["ins"],  N_feat["ins"]+1]
            intervene_Struct.append(1)             
        else:
            layer_szs_ins = None
            intervene_Struct.append(None)

        # Hospital agent policy type
        if policyTypes[1] == 1:
            layer_szs_hosp  = [N_feat["hosp"],  N_feat["hosp"]+1] 
            intervene_Struct.append(1)
        else:
            layer_szs_hosp = None
            intervene_Struct.append(None)

        # Planner agent policy type
        if policyTypes[2] == 1:
            layer_szs_planner = [N_feat["planner"], 4+4+4+4, 4+4+4+4]
            intervene_Struct.append(1)
        else:
            layer_szs_planner  = None
            intervene_Struct.append(None)

        # Combine layer size parameters in list
        layer_szs               = [layer_szs_ins, layer_szs_hosp, layer_szs_planner]
        self.agents             = ["ins", "hosp", "planner"]
        self.layer_szs          = layer_szs
        self.p_elite            = p_elite
        self.intervene_Struct   = intervene_Struct
        self.lmds               = lmds
        self.policyTypes        = policyTypes
        self.opt_type           = opt_type
        self.tracking_path      = tracking_path
        self.N_episodes         = N_episodes
        self.dist_agents        = self.construct_intervention_dictionary(layer_szs, True)
        self.LossTrackers       = self.constructLossTrackers(N_episodes)
        action_sizes            = [2, None, layer_szs_planner[-1]]
        self.ActionTrackers     = self.constructActionTrackers(action_sizes, N_epochs, N_episodes, episode_length-1)
        self.which_agents       = which_agents
        self.scale_consts       = {"ins":1000, "hosp":None, "planner":None}

    def reset_epoch(self):

        # Set up trackers for learning models
        self.ModelTrackers = self.constructModelTrackers()
        self.DictRealizations_N = []

    def reset_episode(self):

        self.reward_t = np.zeros((0,25))
        self.AgentRealization = None

    def update_rewards(self):

        ModelTrackers       = self.ModelTrackers

        # Segement relvant information from reward_t array
        ins_profits             = self.reward_t[:,0]
        geo_tot                 = self.reward_t[:,1:5]
        geo_deceased            = self.reward_t[:,5:9]
        geo_ins                 = self.reward_t[:,9:13]
        geo_illness             = self.reward_t[:,13:17]
        geo_healthy             = self.reward_t[:,17:21]
        geo_terminated_illness  = self.reward_t[:,21:25]
        
        eps=1e-10
        
        # Disease Fatality Rate reward
        mortRts                 = -np.sum(geo_deceased)/np.sum(geo_illness+eps)

        # Incidence Rate reward
        incidenceRts            = -np.sum(geo_illness)/np.sum(geo_healthy+eps)

        # Insured rate reward
        insRts                  = np.sum(geo_ins)/np.sum(geo_tot+eps)

        # Insurance profits reward
        ins_profits             = np.sum(ins_profits)

        # Geo Eqty Mortality rate
        mortRts_geo             = np.sum(geo_deceased,axis=0)/np.sum(geo_terminated_illness+eps,axis=0)
        mortRts_geoEqty         = -np.std(mortRts_geo)

        # Geo Eqty Incidence rate
        incidenceRts_geo        = np.sum(geo_illness,axis=0)/np.sum(geo_healthy+eps,axis=0)
        incidenceRts_geoEqty    = -np.std(incidenceRts_geo)

        # Geo Eqty Insurance rate
        insRts_geo              = np.sum(geo_ins, axis=0)/np.sum(geo_tot+eps,axis=0)
        insRts_geoEqty          = -np.std(insRts_geo)

        # Store model performance information for constructing loss values
        ModelTrackers["mortRts_arr"]                    = np.append(ModelTrackers["mortRts_arr"], mortRts)
        ModelTrackers["insRts_arr"]                     = np.append(ModelTrackers["insRts_arr"], insRts)
        ModelTrackers["incidenceRts_arr"]               = np.append(ModelTrackers["incidenceRts_arr"], incidenceRts)
        ModelTrackers["ins_profits_arr"]                = np.append(ModelTrackers["ins_profits_arr"], ins_profits)
        ModelTrackers["mortRts_geoEqty_arr"]            = np.append(ModelTrackers["mortRts_geoEqty_arr"], mortRts_geoEqty)
        ModelTrackers["insRts_geoEqty_arr"]             = np.append(ModelTrackers["insRts_geoEqty_arr"], insRts_geoEqty)
        ModelTrackers["incidenceRts_geoEqty_arr"]       = np.append(ModelTrackers["incidenceRts_geoEqty_arr"], incidenceRts_geoEqty)

        self.ModelTrackers = ModelTrackers


    def construct_intervention_dictionary(self, layer_szs, train=True, save_path = [],
                                      ins_mu_scale=0, ins_var_scale=2, ins_mu_scale_b=0, ins_var_scale_b=2,
                                      hosp_mu_scale=0, hosp_var_scale=2, hosp_mu_scale_b=0, hosp_var_scale_b=2,
                                      planner_mu_scale=0, planner_var_scale=2, planner_mu_scale_b=0, planner_var_scale_b=2):

        # Import pretrained weights
        if len(save_path) > 0:
            dist_agents = np.load(save_path, allow_pickle=True).item()
            return dist_agents

        # Create distribution parameters used to sample weights of the model
        AgentNames = ['ins', 'hosp', 'planner']
        dist_agents = {'ins':None, 'hosp':None, 'planner':None}
        
        # Get layer sizes for each model
        layer_szs_ins, layer_szs_hosp, layer_szs_planner = layer_szs

        # Initialize agent storage
        for ii in range(len(self.intervene_Struct)):
            if self.intervene_Struct[ii] is not None:
                dist_agents[AgentNames[ii]] = {}

        if layer_szs[0] is not None:
            N_layers_ins        = len(layer_szs[0])
        if layer_szs[1] is not None:
            N_layers_hosp       = len(layer_szs[1])
        if layer_szs[2] is not None:
            N_layers_planner    = len(layer_szs[2])-1

        # Create insurance weights
        if self.intervene_Struct[0] is not None:
            for ii in range(N_layers_ins):

                dist_agents['ins']['ins_mu'+str(ii)]    = np.ones((layer_szs_ins[ii], 1))*ins_mu_scale  # Mean of weights dist
                dist_agents['ins']['ins_var'+str(ii)]   = np.ones((layer_szs_ins[ii], 1))*ins_var_scale # Std of weights dist
                dist_agents['ins']['ins_mu_b'+str(ii)]  = np.ones((1,1))*ins_mu_scale_b                 # Mean of bias dist
                dist_agents['ins']['ins_var_b'+str(ii)] = np.ones((1,1))*ins_var_scale_b                # Std of bias dist


        # Create hospital weights
        if self.intervene_Struct[1] is not None:
            for ii in range(N_layers_hosp):

                dist_agents['hosp']['hosp_mu'+str(ii)]    = np.ones((layer_szs_hosp[ii], 1))*hosp_mu_scale  # Mean of weights dist
                dist_agents['hosp']['hosp_var'+str(ii)]   = np.ones((layer_szs_hosp[ii], 1))*hosp_var_scale # Std of weights dist
                dist_agents['hosp']['hosp_mu_b'+str(ii)]  = np.ones((1,1))*hosp_mu_scale_b                 # Mean of bias dist
                dist_agents['hosp']['hosp_var_b'+str(ii)] = np.ones((1,1))*hosp_var_scale_b                # Std of bias dist
        
        # Create debt forgiveness weights
        if self.intervene_Struct[2] is not None:
            for ii in range(N_layers_planner):

                dist_agents['planner']['planner_mu'+str(ii)]    = np.ones((layer_szs_planner[ii], layer_szs_planner[ii+1]))*planner_mu_scale  # Mean of weights dist
                dist_agents['planner']['planner_var'+str(ii)]   = np.ones((layer_szs_planner[ii], layer_szs_planner[ii+1]))*planner_var_scale # Std of weights dist
                dist_agents['planner']['planner_mu_b'+str(ii)]  = np.ones((1,1))*planner_mu_scale_b                 # Mean of bias dist
                dist_agents['planner']['planner_var_b'+str(ii)] = np.ones((1,1))*planner_var_scale_b                # Std of bias dist

        return dist_agents

    def sample_intervention_parameters(self):

        # Sample the weights of the current model.

        AgentNames = ['ins', 'hosp', 'planner']
        self.AgentRealization = {'ins':None, 'hosp':None, 'planner':None}
        for ii in range(3):                                 # Number of agents

            if self.dist_agents[AgentNames[ii]] is None:
                continue

            CurrDict = self.dist_agents[AgentNames[ii]]     # Current dictionary used for sampling weights
            CurrDict_store  = {}                            # Current dictionary used for storing sampled weights

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

                Params_w = np.random.normal(CurrWeights_mu, CurrWeights_sig)
                Params_b = np.random.normal(CurrWeights_mu_b, CurrWeights_sig_b)

                CurrDict_store[AgentNames[ii]+"_w"+str(jj)] = Params_w
                CurrDict_store[AgentNames[ii]+"_b"+str(jj)] = Params_b

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

        # Get current module's weights
        CurrDict = self.AgentRealization[agent]

        # Apply the weights appropriately
        if agent == 'ins':

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

            ins_premium = X*scale_const

            return ins_premium.flatten()


        # Apply the weights appropriately
        if agent == 'hosp':

            # init params
            M_batch, N_feat = X.shape
            if M_batch > 0:
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

                hosp_scores = X

            else:
                hosp_scores = np.array([])

            return hosp_scores.flatten()


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

            # Subsidy apportioning step
            soft_vals = np.array([np.exp(X[0]), np.exp(X[1]), np.exp(X[2]), np.exp(X[3]),
                              np.exp(X[4]), np.exp(X[5]), np.exp(X[6]), np.exp(X[7]),
                              np.exp(X[8]), np.exp(X[9]), np.exp(X[10]), np.exp(X[11]),
                              np.exp(X[12]), np.exp(X[13]), np.exp(X[14]), np.exp(X[15])])

            X[:4] = soft_vals[:4]/np.sum(soft_vals[:4])
            X[4:8] = soft_vals[4:8]/np.sum(soft_vals[4:8])
            X[8:12] = soft_vals[8:12]/np.sum(soft_vals[8:12])
            X[12:16] = soft_vals[12:16]/np.sum(soft_vals[12:16])
        
            planner_subs = X

            return planner_subs.flatten()


    # Sigmoid function
    def sig(self, x):
        return 1 / (1 + np.exp(-x))
   
    def step(self, weight=0.67):

        tracking_path                   = self.tracking_path
        DictRealizations_N              = self.DictRealizations_N
        mortRts_rewards                 = self.ModelTrackers["mortRts_arr"]
        insRts_rewards                  = self.ModelTrackers["insRts_arr"]
        incidenceRts_rewards            = self.ModelTrackers["incidenceRts_arr"]
        ins_profits_rewards             = self.ModelTrackers["ins_profits_arr"]
        mortRts_geoEqty_rewards         = self.ModelTrackers["mortRts_geoEqty_arr"]
        insRts_geoEqty_rewards          = self.ModelTrackers["insRts_geoEqty_arr"]
        incidenceRts_geoEqty_rewards    = self.ModelTrackers["incidenceRts_geoEqty_arr"]
        LossTrackers                    = self.LossTrackers
        opt_type                        = self.opt_type
        lmds                            = self.lmds
        layer_szs                       = self.layer_szs
        p_elite                         = self.p_elite 

        # Organize for looping

        # Training using cooperation or independent schemes        
        lmd_mortRts                 = lmds["mortRts"]
        lmd_insRts                  = lmds["insRts"]
        lmd_incidenceRts            = lmds["incidenceRts"]
        lmd_ins_profits             = lmds["ins_profits"]
        lmd_mortRts_geoEqty         = lmds["mortRts_geoEqty"]
        lmd_insRts_geoEqty          = lmds["insRts_geoEqty"]
        lmd_incidenceRts_geoEqty    = lmds["incidenceRts_geoEqty"]

        mortRts_rewards_norm                = (mortRts_rewards - np.mean(mortRts_rewards))/np.std(mortRts_rewards)
        insRts_rewards_norm                 = (insRts_rewards - np.mean(insRts_rewards))/np.std(insRts_rewards)
        incidenceRts_rewards_norm           = (incidenceRts_rewards - np.mean(incidenceRts_rewards))/np.std(incidenceRts_rewards)
        ins_profits_rewards_norm            = (ins_profits_rewards - np.mean(ins_profits_rewards))/np.std(ins_profits_rewards)
        mortRts_geoEqty_rewards_norm        = (mortRts_geoEqty_rewards - np.mean(mortRts_geoEqty_rewards))/np.std(mortRts_geoEqty_rewards)
        insRts_geoEqty_rewards_norm         = (insRts_geoEqty_rewards - np.mean(insRts_geoEqty_rewards))/np.std(insRts_geoEqty_rewards)
        incidenceRts_geoEqty_rewards_norm   = (incidenceRts_geoEqty_rewards - np.mean(incidenceRts_geoEqty_rewards))/np.std(incidenceRts_geoEqty_rewards)

        coop_rewards     = (lmd_mortRts*mortRts_rewards_norm + lmd_insRts*insRts_rewards_norm + \
                            lmd_incidenceRts*incidenceRts_rewards_norm + lmd_ins_profits*ins_profits_rewards_norm + \
                            lmd_mortRts_geoEqty*mortRts_geoEqty_rewards_norm + lmd_insRts_geoEqty*insRts_geoEqty_rewards_norm + \
                            lmd_incidenceRts_geoEqty*incidenceRts_geoEqty_rewards_norm)/7

        ind_rewards = [ins_profits_rewards_norm, mortRts_rewards_norm, coop_rewards]
        if opt_type == 'independent':
            train_rewards   = ind_rewards.copy()
        else:
            train_rewards   = [coop_rewards, coop_rewards, coop_rewards]

        # Initialize
        names = ['ins', 'hosp', 'planner']
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
        names        = ['mortRts', 'insRts', 'incidenceRts', 'ins_profits', 'mortRts_geoEqty', 'insRts_geoEqty', 'incidenceRts_geoEqty']
        ind_rewards  = [mortRts_rewards, insRts_rewards, incidenceRts_rewards, ins_profits_rewards, mortRts_geoEqty_rewards, insRts_geoEqty_rewards, incidenceRts_geoEqty_rewards]

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
        "mortRts_rewards_arr":                      np.zeros((0,N_episodes)),
        "insRts_rewards_arr":                       np.zeros((0,N_episodes)),
        "incidenceRts_rewards_arr":                 np.zeros((0,N_episodes)),
        "ins_profits_rewards_arr":                  np.zeros((0,N_episodes)),
        "mortRts_geoEqty_rewards_arr":              np.zeros((0,N_episodes)),
        "insRts_geoEqty_rewards_arr":               np.zeros((0,N_episodes)),
        "incidenceRts_geoEqty_rewards_arr":         np.zeros((0,N_episodes)),

        "mortRts_mean_rewards":                     np.zeros((0)),
        "insRts_mean_rewards":                      np.zeros((0)),
        "incidenceRts_mean_rewards":                np.zeros((0)),
        "ins_profits_mean_rewards":                 np.zeros((0)),
        "mortRts_geoEqty_mean_rewards":             np.zeros((0)),
        "insRts_geoEqty_mean_rewards":              np.zeros((0)),
        "incidenceRts_geoEqty_mean_rewards":        np.zeros((0)),

        }

        return LossTrackers


    def constructModelTrackers(self):

        ModelTrackers = {
            "mortRts_arr":              np.array([]),
            "insRts_arr":               np.array([]),
            "incidenceRts_arr":         np.array([]),
            "ins_profits_arr":          np.array([]),
            "mortRts_geoEqty_arr":      np.array([]),
            "insRts_geoEqty_arr":       np.array([]),
            "incidenceRts_geoEqty_arr": np.array([])
            }

        return ModelTrackers


    def constructActionTrackers(self, action_szs, N_epochs, N_episodes, episode_len):

        ActionTrackers = {}

        if action_szs[0] is not None:
            ActionTrackers["ins_actions"] = np.zeros((N_epochs, N_episodes, episode_len, action_szs[0]))

        if action_szs[1] is not None:
            ActionTrackers["hosp_actions"] = np.zeros((N_epochs, N_episodes, episode_len, action_szs[1]))

        if action_szs[2] is not None:
            ActionTrackers["planner_actions"] = np.zeros((N_epochs, N_episodes, episode_len, action_szs[2]))

        return ActionTrackers

