import numpy as np
from multiagent.core_healthcare import World, AgentIns, AgentHosp, AgentPlanner, AllPopulationInformation, TrackingIndicators
import os
import pandas as pd
import pickle
from itertools import compress

''' This script constructs the Healthcare MAFE class used to sample observations from actions made by agents'''
class Environment(object):

    def __init__(self, policyTypes, which_agents, k_cycle, episode_length):
        
        self.world = self.make_world(policyTypes, which_agents, k_cycle, episode_length)
    
    def make_world(self, policyTypes, which_agents, k_cycle, episode_length):

        world = World()
        
        # Set any world properties first
        world.agents = []

        world.agents.append(AgentIns())
        world.agents[0].policyType = policyTypes[0]
        if which_agents[0]:
            world.agents[0].agent_flag = True
        else:
            world.agents[0].agent_flag = False

        world.agents.append(AgentHosp())
        world.agents[1].policyType = policyTypes[1]
        if which_agents[1]:
            world.agents[1].agent_flag = True            
        else:
            world.agents[1].agent_flag = False            

        world.agents.append(AgentPlanner())
        world.agents[2].policyType = policyTypes[2]
        if which_agents[2]:
            world.agents[2].agent_flag = True
        else:
            world.agents[2].agent_flag = False

        world.allpopulationinformation  = AllPopulationInformation()
        world.trackingindicators        = TrackingIndicators()
        world.k_cycle=k_cycle                                                      # Cyclicity of actions
        world.N_geog = 4	                                                   # Number of geographic regions
        world.dist_init_beds = np.ones((world.N_geog,1),dtype=int)*100             # Initial condition: number of hospital beds in each geographic region
        world.feature_increment = 10                                               # How much of an improvement can we make in a feature at any given point in time
        world.store_path = "env_dta/"                                              # Path to stored data used to construct features
        world.save_dta_flag = False                                                 
        world.hosp_bill = 10000                                                    # Cost per hospital visit
        world.episode_length = episode_length                                      # Number of time steps in episode

        # Import data and models that are part of the environment
        world = self.loadDtaModels(world)

        return world

    def loadDtaModels(self, world):

        # Import needed world variables
        k_cycle             = world.k_cycle
        hosp_bill           = world.hosp_bill
        feature_increment   = world.feature_increment
        save_dta_flag       = world.save_dta_flag

        # Read in data from dataset
        print("Loading dataset...")
        if save_dta_flag:
            df_train, df_test, X_train, X_test, feature_names, feature_names_preproc, norm_params, feature_start_idxs, feature_extremes, feature_polarity = load_healthcare_data()
            reg = self.train_health_regressor(df_train, X_train, norm_params, feature_start_idxs)
            pickle.dump(reg, open("./env_dta/health_regressor.pkl", 'wb'))
            np.savetxt("./env_dta/X_train.txt", X_train)
            df_train.to_pickle("./env_dta/df_train.pkl")
            np.savetxt("./env_dta/X_test.txt", X_test)
            df_test.to_pickle("./env_dta/df_test.pkl")
            np.save("./env_dta/feature_names.npy", feature_names)
            np.save("./env_dta/feature_names_preproc.npy", feature_names_preproc)
            np.save("./env_dta/norm_params.npy", norm_params)
            np.save("./env_dta/feature_start_idxs.npy", feature_start_idxs)
            np.save("./env_dta/feature_extremes.npy",feature_extremes)
            np.save("./env_dta/feature_polarity.npy",feature_polarity)
    
        else:
            reg                     = pickle.load(open("./env_dta/health_regressor_12122024.pkl", 'rb'))
            X_train                 = np.loadtxt("./env_dta/X_train.txt")
            df_train                = pd.read_pickle("./env_dta/df_train.pkl")
            X_test                  = np.loadtxt("./env_dta/X_test.txt")
            df_test                 = pd.read_pickle("./env_dta/df_test.pkl")
            feature_names           = np.load("./env_dta/feature_names.npy").tolist()
            feature_names_preproc   = np.load("./env_dta/feature_names_preproc.npy").tolist()
            norm_params             = np.load("./env_dta/norm_params.npy", allow_pickle=True).item()
            feature_start_idxs      = np.load("./env_dta/feature_start_idxs.npy", allow_pickle=True).item()
            feature_extremes        = np.load("./env_dta/feature_extremes.npy",allow_pickle=True).item()
            feature_polarity        = np.load("./env_dta/feature_polarity.npy",allow_pickle=True).item()

        # Import model information to the world
        world.feature_start_idxs    = feature_start_idxs
        world.feature_extremes      = feature_extremes
        world.norm_params           = norm_params
        world.feature_polarity      = feature_polarity
        world.feature_names         = feature_names
        world.feature_names_preproc = feature_names_preproc


        # Final parameter initialization
        world.init_params =     {"cycle_values":k_cycle,                                              # Storing step counts between actions for cyclic and noncyclic actions
                                 "Planner_budget":2.5e8,                                              # Amount of budget planner starts with
                                 "Planner_budget_update":2.5e8,                                       # Amount of budget replenished every k_cycle['planner'] time steps
                                 "health_regressor":reg,                                              # regressor for predicting person's health
                                 "params_stillill_outcome": np.array([0, 1.03, -7, 0, -0, -0]),       # Parameters still ill update rule
                                 "params_deceased_outcome": np.array([1.96, -1.02, 2, 3, 3, -7]),     # Parameters deceased update rule
                                 "const_div_HIPCOST":1000,                                            # normalizing constant for price of health insurance paid by customer
                                 "const_div_HIPFULLCOST":1000,                                        # normalizing constant for full price of health insurance (paid by customer + subsidy)
                                 "const_div_NGEOBED":100,                                             # normalizing constant for number of beds in a geographic area
                                 "const_div_HOSPCOST":hosp_bill,                                      # normalizing constant for HOSPCOST features
                                 "const_div_WAITBED":10,                                              # normalizing constant for WAITBED features
                                 "const_div_ILLTIME":10,                                              # normalizing constant for ILLTIME features
                                 "const_div_PLANBUDGET": 2.5e8,                                       # normalizing constant for PLANBUDGET features
                                 "feature_increment":feature_increment,                               # How much of an improvement can we make in a feature at any given point in time
                                 "feature_start_idxs":feature_start_idxs,                             # Contains information about the column indicies in X taken up by a given feature after binarization.
                                 "feature_extremes":feature_extremes,                                 # min and max values for each feature
                                 "norm_params":norm_params,                                           # Parameters for normalizing/unnormalizing different features
                                 "updateable_feature": list(feature_polarity.keys()),                 # Features that may change as a result of environmental factors
                                 "feature_polarity": feature_polarity,                                # Polarity of updateable features (sign indicates whether increasing the value results in feature improvement/deterioration)
                                 "feature_names":feature_names,                                       # Names of features in df (original)
                                 "feature_names_preproc":feature_names_preproc,                       # Names of features in X  (after binarizing categorical variables)
                                 "hosp_bill": hosp_bill,                                              # Hospital bill for the whole stay
                                 "time_delay_pub_invest":12,                                           
                                 "time_delay_insurance_subsidy":0,                                    
                                 "time_coeffs_bed_construct":[0.05, 2],                               # Baseline, proportional parameters for time to hospital build infrastructure
                                 "deterioration_public":0,                                            # Used to apply deterioration to population features (i.e. if never any public health investment, the population health deteriorates
                                 "deterioration_hosp_infra":2e6,                                      # Used to apply deterioriation to hospital infrastructure each time step
                                 "hospital_bed_cost_params":[3e7, 1e6]}                               # Baseline cost, proportional cost per unit to build hospital infrastructure

        # Augment data from dataset with environemental tracker features (e.g. who is in the hospital/how long in hospital/how long sick/etc.)
        X_pop, df_pop, feature_names, feature_names_preproc, norm_params, feature_start_idxs = self.augment_features(X_test,
                                                                                                        df_test,
                                                                                                        feature_names,
                                                                                                        feature_names_preproc,
                                                                                                        norm_params,
                                                                                                        feature_start_idxs,
                                                                                                        world.init_params)
        # Function to introduce bias goes here
        X_pop, df_pop = self.add_HEALTH_bias(X_pop, df_pop, reg, world.N_geog, feature_start_idxs, feature_extremes)
        
        world.allpopulationinformation.reg                      = reg
        world.allpopulationinformation.X_pop                    = X_pop
        world.allpopulationinformation.df_pop                   = df_pop
        world.allpopulationinformation.feature_names            = feature_names
        world.allpopulationinformation.feature_names_preproc    = feature_names_preproc
        world.allpopulationinformation.norm_params              = norm_params
        world.allpopulationinformation.feature_start_idxs       = feature_start_idxs
        world.allpopulationinformation.feature_extremes         = feature_extremes
        world.allpopulationinformation.feature_polarity         = feature_polarity

        return world

    # Do this before each episode starts
    def reset_world(self):
        
        
        # Set up trackers for environment and learning models
        self.world.trackingindicators.EnvTrackers = self.constructEnvTrackers()

        # Initialize all State/Feature information
        self.init_agent_features()

        # Get first observation to return
        obs = [self.world.agents[0].state, self.world.agents[1].state, self.world.agents[2].state]

        return obs


    def step(self, iter_num):

        return self.world.step(iter_num)

    def add_HEALTH_bias(self, X, df, reg, N_geog, feature_start_idxs, feature_extremes):	

        bias_n_g = np.array([1.        , 2.33333333, 3.66666667, 5.        ])
        df_bias = df.copy().drop(df.index)
        X_bias = np.zeros((0,X.shape[1]))
        N_features = X.shape[1]
        delete_features = N_features-np.linspace(1,11,11, dtype=int)
        delete_features = np.append(delete_features, feature_start_idxs['HEALTH'][1])
        N_tot = int(X.shape[0]/4)

        for n_g in range(N_geog):

            X_geog  = X[df["REGION"]==n_g]
            df_geog = df[df["REGION"]==n_g]
            N_choice = X_geog.shape[0]
            idx_choices = np.linspace(0, N_choice-1, N_choice, dtype=int)

            X_wo_HEALTH = np.delete(X_geog, delete_features, axis=1)
            x_health = reg.predict(X_wo_HEALTH)
            x_health = np.clip(x_health, feature_extremes["HEALTH"][0], feature_extremes["HEALTH"][1])
            dist = (4.15-abs(x_health-bias_n_g[n_g]))**4
            dist = dist/np.sum(dist)
            sample_idxs = np.random.choice(idx_choices, N_tot, replace=True, p= dist)
      
            df_geog     = df_geog.iloc[sample_idxs]
            X_geog_new  = X_geog[sample_idxs]

            df_bias = pd.concat([df_bias, df_geog], ignore_index=True)
            X_bias = np.vstack((X_bias, X_geog_new))

        # Shuffle rows of new data
        N     = X_bias.shape[0]
        idxs  = np.linspace(0,N-1,N, dtype=int)
        np.random.shuffle(idxs)

        df_bias = df_bias.iloc[idxs]
        X_bias  = X_bias[idxs]

        # Reset indices of data frames
        df_bias = df_bias.reset_index(drop=True)

        return  X_bias, df_bias  

    def train_health_regressor(self, df, X, norm_params, feature_start_idxs):

        # Train regressor to predict a person's health given their other features
    
        y_train = df['HEALTH'].to_numpy()
        X_train = np.delete(X, feature_start_idxs['HEALTH'][1], axis=1)
        reg = LinearRegression().fit(X_train, y_train)
        reg.coef_[44] = -reg.coef_[44]*10
        reg.coef_[54] = -reg.coef_[54]*20

        return reg

    def augment_features(self, X_pop, df_pop, feature_names, feature_names_preproc, norm_params, feature_start_idxs, init_params):

        N = df_pop.shape[0]

        # Augment feature_start_idxs
        df_sz = df_pop.shape[1]
        X_sz = X_pop.shape[1]
        feature_start_idxs["NEEDBED"]       = (df_sz,    X_sz,    X_sz+1)
        feature_start_idxs["INHOSP"]        = (df_sz+1,  X_sz+1,  X_sz+2)
        feature_start_idxs["ILLNESS"]       = (df_sz+2,  X_sz+2,  X_sz+3)
        feature_start_idxs["DECEASED"]      = (df_sz+3,  X_sz+3,  X_sz+4)
        feature_start_idxs["NGEOBED"]       = (df_sz+4,  X_sz+4,  X_sz+5)
        feature_start_idxs["HIPCOST"]       = (df_sz+5,  X_sz+5,  X_sz+6)
        feature_start_idxs["HIPFULLCOST"]   = (df_sz+6,  X_sz+6,  X_sz+7)
        feature_start_idxs["HOSPCOST"]      = (df_sz+7,  X_sz+7,  X_sz+8)
        feature_start_idxs["WAITBED"]       = (df_sz+8,  X_sz+8,  X_sz+9)
        feature_start_idxs["ILLTIME"]       = (df_sz+9,  X_sz+9,  X_sz+10)
        feature_start_idxs["PLANBUDGET"]    = (df_sz+10, X_sz+10, X_sz+11)

        # Augment df
        df_pop['NEEDBED']       = np.ones(N,dtype=bool)*False
        df_pop['INHOSP']        = np.ones(N,dtype=bool)*False
        df_pop['ILLNESS']       = np.ones(N,dtype=bool)*False
        df_pop['DECEASED']      = np.ones(N,dtype=bool)*False
        df_pop['NGEOBED']       = np.zeros(N)
        df_pop['HIPCOST']       = np.zeros(N)
        df_pop['HIPFULLCOST']   = np.zeros(N)
        df_pop['HOSPCOST']      = np.zeros(N)
        df_pop['WAITBED']       = np.zeros(N)
        df_pop['ILLTIME']       = np.zeros(N)
        df_pop['PLANBUDGET']    = np.zeros(N)
        df_pop = df_pop.astype('object')
        df_pop = df_pop.reset_index(drop=True)

        # Augment X
        zero  = np.zeros((N,1))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))

        # Update feature_names
        feature_names.append('NEEDBED')
        feature_names.append('INHOSP')
        feature_names.append('ILLNESS')
        feature_names.append('DECEASED')
        feature_names.append('NGEOBED')
        feature_names.append('HIPCOST')
        feature_names.append('HIPFULLCOST')
        feature_names.append('HOSPCOST')
        feature_names.append('WAITBED')
        feature_names.append('ILLTIME')
        feature_names.append('PLANBUDGET')

        # Update feature_names_preproc
        feature_names_preproc.append('NEEDBED')
        feature_names_preproc.append('INHOSP')
        feature_names_preproc.append('ILLNESS')
        feature_names_preproc.append('DECEASED')
        feature_names_preproc.append('NGEOBED')
        feature_names_preproc.append('HIPCOST')
        feature_names_preproc.append('HIPFULLCOST')
        feature_names_preproc.append('HOSPCOST')
        feature_names_preproc.append('WAITBED')
        feature_names_preproc.append('ILLTIME')
        feature_names_preproc.append('PLANBUDGET')

        # Update normalization parameters
        const_div = [[]]*len(norm_params['mu_test'])
        const_div.append([])                                    # Binary for NEEDBED
        const_div.append([])                                    # Binary for INHOSP
        const_div.append([])                                    # Binary for ILLNESS
        const_div.append([])                                    # Binary for DECEASED
        const_div.append(init_params["const_div_NGEOBED"])      # Contiuous for NGEOBED.     Approx guess of max max amount of beds in region for normalizing (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_HIPCOST"])      # Contiuous for HIPCOST.     Approx guess of max max amount of a health insurance premium in region for normalizing (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_HIPFULLCOST"])  # Contiuous for HIPFULLCOST. Approx guess of max max amount paid to insurance in region (subsidized and non-subsidized combined) for normalizing (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_HOSPCOST"])     # Contiuous for HOSPCOST.    Approx guess of max  hospital cost for bed (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_WAITBED"])      # Contiuous for WAITBED.     Approx guess of max time waiting for bed (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_ILLTIME"])      # Contiuous for ILLTIME.     Approx guess of max time someone is sick (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_PLANBUDGET"])   # Contiuous for PLANBUDGET.  Approx guess of max size of planner budget (division by this should hopefully mean max time is [0,1])
        extension = [[]]*(len(const_div)-len(norm_params['mu_test']))
        norm_params['mu_train'].extend(extension)
        norm_params['mu_test'].extend(extension)
        norm_params['sig_train'].extend(extension)
        norm_params['sig_test'].extend(extension)
        norm_params['const_div'] = const_div

        return X_pop, df_pop, feature_names, feature_names_preproc, norm_params, feature_start_idxs


    def constructEnvTrackers(self):

        init_params     = self.world.init_params
        N_geog          = self.world.N_geog
        dist_init_beds  = self.world.dist_init_beds

        if isinstance(dist_init_beds,int):
            dist_init_beds = np.ones((N_geog,1),dtype=int)*dist_init_beds

        # Each row of array contains number of projects funded in each region.
        # E.g. np.array([5,10]) means two projects funded in region, first producing 5 beds and second producing 10.
        constructing_beds = [np.array([],dtype=int)]*N_geog

        # Time left to construct beds
        constructing_time = [np.array([],dtype=int)]*N_geog

        # How many people waiting for beds in each location
        queue_beds = np.zeros((N_geog,0))

        EnvTrackers = {
            "Geographic_current_queue_beds": queue_beds,
            "Geographic_current_total_beds": dist_init_beds.copy(),
            "Geographic_current_available_beds": dist_init_beds.copy(),
            "Geographic_constructing_beds": constructing_beds,
            "Geographic_constructing_time": constructing_time,
            "Geographic_mortalities": np.zeros((N_geog,0)),
            "Geographic_relative_ins_cost": [],
            "N_geog":N_geog,
            "actions":{"ins":None, "hosp":None, "planner":None},
            "cycle_values":init_params["cycle_values"],
            "Planner_budget": init_params["Planner_budget"],
            "Planner_budget_update": init_params["Planner_budget_update"],
            "health_regressor": init_params["health_regressor"],
            "params_stillill_outcome":init_params["params_stillill_outcome"],
            "params_deceased_outcome":init_params["params_deceased_outcome"],
            "updateable_feature": init_params["updateable_feature"],
            "const_div_HIPCOST":init_params["const_div_HIPCOST"],
            "const_div_HIPFULLCOST":init_params["const_div_HIPFULLCOST"],
            "const_div_NGEOBED":init_params["const_div_NGEOBED"],
            "const_div_HOSPCOST":init_params["const_div_HOSPCOST"],                 
            "const_div_WAITBED":init_params["const_div_WAITBED"],                             
            "const_div_ILLTIME":init_params["const_div_ILLTIME"],
            "const_div_PLANBUDGET":init_params["const_div_PLANBUDGET"],
            "feature_start_idxs":init_params["feature_start_idxs"],
            "feature_increment":init_params["feature_increment"],
            "feature_extremes":init_params["feature_extremes"],
            "norm_params":init_params["norm_params"],
            "feature_names":init_params["feature_names"].copy(),
            "feature_names_preproc":init_params["feature_names_preproc"].copy(),
            "feature_polarity":init_params["feature_polarity"],
            "hospital_bed_cost_params": init_params["hospital_bed_cost_params"],
            "time_delay_pub_invest":init_params["time_delay_pub_invest"],
            "time_delay_insurance_subsidy":init_params["time_delay_insurance_subsidy"],
            "time_coeffs_bed_construct":init_params["time_coeffs_bed_construct"],
            "deterioration_public":init_params["deterioration_public"],
            "deterioration_hosp_infra":init_params["deterioration_hosp_infra"],
            "hosp_bill": init_params['hosp_bill'],
            "Insurance_profits_track":np.array([]),  
            "Geographic_outpocket_hosp_expenses_track":np.zeros((N_geog,0)),                    # Total out-of-pocket expenses for people discharged from hospital at given time step for each geolocation in the system
            "Geographic_payments_to_ins_track":np.zeros((N_geog,0)),                            # Total payments made to insurance by individuals in each geographic region
            "Geographic_number_discharged_track":np.zeros((N_geog,0)),                          # Total people discharged from hospital at given time step for each geolocation in the system
            "Geographic_number_ill_track":np.zeros((N_geog,0)),                                 # Total people ill at given time step for each geolocation in the system
            "Geographic_number_deceased_track":np.zeros((N_geog,0)),                            # Total deceased at given time step for each geolocation in the system
            "Geographic_number_total_beds_track":np.zeros((N_geog,0)),                          # Total beds (available+unavailable) at given time step for each geolocation in the system
            "Geographic_number_insured_track":np.zeros((N_geog,0)),                             # Total people insured at given time step for each geolocation in the system
            "Geographic_number_uninsured_track":np.zeros((N_geog,0)),                           # Total people uninsured at given time step for each geolocation in the system
            "Geographic_number_total_track":np.zeros((N_geog,0)),                               # Total people at given time step for each geolocation in the system
            "Geographic_total_number_want_bed_track":np.zeros((N_geog,0)),                      # Total people who want hosp bed at given time step for each geolocation in the system
            "Geographic_wait_times_admitted_track":np.zeros((N_geog,0)),                        # Average time waited for people granted hospital bed at given time step for each geolocation in the system 
            "Geographic_number_given_bed_track":np.zeros((N_geog,0)),                           # Total people given hospital bed at given time step for each geolocation in the system
            "Geographic_number_healthy_track":np.zeros((N_geog,0)),
            "Geographic_number_terminated_illness_track":np.zeros((N_geog,0)),
            "rewards":np.zeros((0,6)),
            "Ins_money_made":None
            }
        return EnvTrackers 


    def init_agent_features(self):


        # Import needed information from the world
        X_pop       = self.world.allpopulationinformation.X_pop.copy()
        df_pop      = self.world.allpopulationinformation.df_pop.copy()
        EnvTrackers = self.world.trackingindicators.EnvTrackers

        # TODO: look for any specific features to remove/be filtered once this whole thing is up and running and we are honing in on details of the system

        feature_names           =   EnvTrackers["feature_names"]
        feature_names_preproc   =   EnvTrackers["feature_names_preproc"]
        norm_params             =   EnvTrackers["norm_params"]
        feature_start_idxs      =   EnvTrackers["feature_start_idxs"]

        # Initialize HEALTH features for everyone in the system
        df_pop, X_pop = self.world.HealthUpdate(df_pop, X_pop, EnvTrackers)

        # Insurance state/feature information
        remove_feat_idxs            = np.array([var[:2]=="HI" or var=="PLANBUDGET" for var in feature_names ])
        remove_feat_idxs_preproc    = np.array([var[:2]=="HI" or var=="PLANBUDGET" for var in feature_names_preproc ])
        feat_remove                 = list(compress(feature_names, remove_feat_idxs))
        df_ins                      = df_pop.drop(columns=feat_remove)
        X_ins                       = X_pop[:,~remove_feat_idxs_preproc]

        # Hospital state/feature information
        idx_need_bed                    = df_pop['NEEDBED'] == True
        df_hosp                         = df_pop[idx_need_bed]
        X_hosp                          = X_pop[idx_need_bed]
        remove_feat_idxs                = np.array([var=="PLANBUDGET" for var in feature_names         ])
        remove_feat_idxs_preproc        = np.array([var=="PLANBUDGET" for var in feature_names_preproc ])
        feat_remove                     = list(compress(feature_names, remove_feat_idxs))
        df_hosp                         = df_hosp.drop(columns=feat_remove)
        X_hosp                          = X_hosp[:, ~remove_feat_idxs_preproc]

        # Omni. planner state/feature information
        df_planner  =  df_pop
        X_planner   =  X_pop

        X_full = {"global":X_pop, "ins":X_ins, "hosp":X_hosp, "planner":X_planner}
        df_full = {"global":df_pop, "ins":df_ins, "hosp":df_hosp, "planner":df_planner}

        EnvTrackers["feature_names"]         = feature_names
        EnvTrackers["feature_names_preproc"] = feature_names_preproc
        EnvTrackers["norm_params"]           = norm_params
        EnvTrackers["feature_start_idxs"]    = feature_start_idxs

        self.world.allpopulationinformation.X_full  =   X_full
        self.world.allpopulationinformation.df_full =   df_full
        self.world.agents[0].state                  =   X_ins
        self.world.agents[1].state                  =   X_hosp
        self.world.agents[2].state                  =   X_planner
        self.world.trackingindicators.EnvTrackers   =   EnvTrackers 

    def constructLossTrackers(self):

        LossTrackers = {
        "ins_rewards_arr":      np.array([]),
        "hosp_rewards_arr":     np.array([]),
        "planner_rewards_arr":  np.array([]),
        "mortRts_geoEqty_arr":  np.array([]),
        "insRts_geoEqty_arr":   np.array([])
        }

        return LossTrackers

























            
