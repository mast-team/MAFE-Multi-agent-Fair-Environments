import numpy as np
import pandas as pd
import random
from numpy.random import  binomial, uniform
from itertools import compress

class AllPopulationInformation(object):
    def __init__(self): 
        
        # Population feature information
        self.df_full                = None
        self.X_full                 = None
        self.X_pop                  = None
        self.df_pop                 = None
        self.reg                    = None
        self.feature_names          = None
        self.feature_names_preproc  = None
        self.norm_params            = None
        self.feature_start_idxs     = None
        self.feature_extremes       = None
        self.feature_polarity       = None

class TrackingIndicators(object):
    def __init__(self): 

        # Environment indicators to monitor every round
        self.EnvTrackers = None  # All parameters and things to be tracked inside the environment
        
# Properties of agent entities
class AgentIns(object):
    def __init__(self):

        # agent action structure
        self.policyType = None
                        
        # state
        self.state = None
        
        # action
        self.action = None

        # subsidy
        self.subsidy = None

        # Flags where agent is an RL model (True) or part of the environment (False)
        self.agent_flag = None
        
# Properties of agent entities
class AgentHosp(object):
    def __init__(self):

        # agent action structure
        self.policyType = None
        
        # state
        self.state = None
        
        # action
        self.action = None

        # subsidy
        self.subsidy = None

        # Flags where agent is an RL model (True) or part of the environment (False)
        self.agent_flag = None

# Properties of agent entities
class AgentPlanner(object):
    def __init__(self):

        # agent action structure
        self.policyType = None
        
        # state 
        self.state              = None
        
        # action
        self.action             = None

        # subsidy (for public)
        self.public_subsidy     = None

        # subsidy (rollover)
        self.rollover_subsidy   = None

        # Flags where agent is an RL model (True) or part of the environment (False)
        self.agent_flag         = None

# Multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)

        # Object classes in the simulation
        self.agents                     = None # Each entry is an agent in the simulation 
        self.allpopulationinformation   = None # All population related information
        self.trackingindicators         = None

        # Parameters of the simulation
        self.k_cycle                = None          
        self.save_dta_flag          = None
        self.N_geog                 = None	                                                           
        self.dist_init_beds         = None                         
        self.feature_increment      = None                                             
        self.agent_config           = None                                                     
        self.store_path             = None
        self.hosp_bill              = None
        self.episode_length         = None
        self.init_params            = None
        
    # Sigmoid function
    def sig(self, x):
        return 1 / (1 + np.exp(-x))
    
    # update state of the world
    def step(self, iter_num):
        
        # ------------------Apply Decisions-----------------
        # Apply Omniscient Planner Agent decisions
        self.apply_planner_actions()

        # Apply Insurance Agent decisions
        self.apply_ins_actions()

        # Apply Hospital Agent decisions
        self.apply_hosp_actions()

        # Update environment (or do this in Env script?) (This should handle queueing stuff an swapping out agents to be subsumed into the environment)
        failed_ep = self.update_env()

        # ------------------Regular State Information-----------------
        # Update Insurance Agent state
        self.update_ins_agent_state(iter_num)
        state0 = self.agents[0].state

        # Update Hospital Agent state
        self.update_hosp_agent_state(iter_num)
        state1 = self.agents[1].state

        # Update Omniscient Agent state
        self.update_planner_agent_state(iter_num)
        state2 = self.agents[2].state

        # Things to return
        obs = [state0, state1, state2]
        reward = self.trackingindicators.EnvTrackers["rewards"]

        # End of episode
        if failed_ep or iter_num==self.episode_length:
            dones = 1
        else:
            dones = 0

        return obs, reward, failed_ep, dones


    def update_env(self):

        EnvTrackers     = self.trackingindicators.EnvTrackers
        df_full         = self.allpopulationinformation.df_full
        X_full          = self.allpopulationinformation.X_full

        # Grab global variables
        df_global         = df_full["global"]
        X_global          = X_full["global"]

        # Update people who are healthy alive/sick alive/dead
        df_global, X_global, EnvTrackers = self.IllnessUpdates(df_global, X_global, EnvTrackers)

        # Update insurance profits (total amount paid by all customers - hospital bills)
        EnvTrackers = self.UpdateInsuranceProfits(df_global, EnvTrackers)

        # Get Rewards
        EnvTrackers = self.UpdateRewardsTrackers(EnvTrackers)

        # (Environment update) Update peron's  features and reinitialize them in the population (After death for one person)
        X_global, df_global, EnvTrackers = self.ReplenishPop(X_global, df_global, EnvTrackers)

        # Check if failed episode
        failed_ep = self.checkFailedEpisode(df_full, X_full, EnvTrackers)

        # Store back
        df_full["global"]                   = df_global
        X_full["global"]                    = X_global

        self.allpopulationinformation.X_full   = X_full
        self.allpopulationinformation.df_full  = df_full
        self.trackingindicators.EnvTrackers = EnvTrackers

        return failed_ep

    def ReplenishPop(self, X, df, EnvTrackers):

        norm_params = EnvTrackers["norm_params"]
        feature_start_idxs = EnvTrackers["feature_start_idxs"]

        # Population Rebirth
        idx_done_in_env                     = df["DECEASED"] == 1
        df["DECEASED"][idx_done_in_env]     = 0
        X[idx_done_in_env, feature_start_idxs['DECEASED'][1]]   = 0

        return X, df, EnvTrackers

    def apply_planner_actions(self):

        agents              = self.agents
        N_geog              = self.N_geog
        EnvTrackers         = self.trackingindicators.EnvTrackers
        action_planner      = agents[2].action
        norm_params         = EnvTrackers["norm_params"]
        feature_start_idxs  = EnvTrackers["feature_start_idxs"]
        df_full             = self.allpopulationinformation.df_full
        X_full              = self.allpopulationinformation.X_full

        # Grab global variables
        df_global         = df_full["global"]
        X_global          = X_full["global"]

        if action_planner is not None:

            # Get subsidy actions for planner
            subsidies_provided = action_planner # percentages that sum to 1

            # Break down and restructure subsidies        
            which_subsidy = ["rollover", "ins", "hosp", "public"]
            each_subsidy    = {"rollover":0, "ins":np.array([]), "hosp":np.array([]), "public":np.array([])}

            # Percentage of budget to rollover to next time step
            each_subsidy["rollover"] = subsidies_provided[0]*EnvTrackers["Planner_budget"]

            # Percentage of budget disbursed throughout current time step
            for n_subsidy in range(3):
                which_subsidy_percent = subsidies_provided[n_subsidy+1]
                curr_subsidy_name = which_subsidy[n_subsidy+1]
                for n_g in range(N_geog):
                    subsidy_percent_geog = which_subsidy_percent*subsidies_provided[int(n_subsidy*N_geog+n_g+4)]
                    curr_subsidy = subsidy_percent_geog*EnvTrackers["Planner_budget"]
                    each_subsidy[curr_subsidy_name] = np.append(each_subsidy[curr_subsidy_name], curr_subsidy)
        else:
            each_subsidy = {"rollover":None, "ins":None, "hosp":None, "public":None}

        agents[0].ins_subsidy      = each_subsidy["ins"]
        agents[1].hosp_subsidy     = each_subsidy["hosp"]
        agents[2].public_subsidy   = each_subsidy["public"]
        agents[2].rollover_subsidy = each_subsidy["rollover"]

        # Update planner's budget for next round
        if action_planner is not None:
            EnvTrackers["Planner_budget"]                       = EnvTrackers["Planner_budget"] + each_subsidy["rollover"]
            df_global["PLANBUDGET"]                             = np.ones(df_global.shape[0])*EnvTrackers["Planner_budget"]
            X_global[:,feature_start_idxs["PLANBUDGET"][1]]     = df_global["PLANBUDGET"]/norm_params["const_div"][feature_start_idxs["PLANBUDGET"][0]]

        # Apply public subsidy
        if action_planner is not None:
            df_global, X_global, EnvTrackers = self.UpdatePopulationFeatures(df_global, X_global, EnvTrackers, agents)

        # Update hospital infrastructure
        df_global, X_global, EnvTrackers, agents = self.UpdateHospitalInfrastructure(df_global, X_global, agents, EnvTrackers)

        # Store back
        df_full["global"]                   = df_global
        X_full["global"]                    = X_global

        self.allpopulationinformation.X_full   = X_full
        self.allpopulationinformation.df_full  = df_full
        self.agents                             = agents
        self.trackingindicators.EnvTrackers     = EnvTrackers

    def update_planner_agent_state(self, t):

        agents          = self.agents
        df_full         = self.allpopulationinformation.df_full
        X_full          = self.allpopulationinformation.X_full
        EnvTrackers     = self.trackingindicators.EnvTrackers

        # Update State Information Omin. Planner
        df_full, X_full = self.State_Update(df_full, X_full, EnvTrackers, "planner", t+1, self.k_cycle["planner"])
        agents[2].state = X_full["planner"]

        self.agents                             = agents
        self.allpopulationinformation.df_full   = df_full
        self.allpopulationinformation.X_full    = X_full  

    def apply_ins_actions(self):

        agents          = self.agents
        EnvTrackers     = self.trackingindicators.EnvTrackers
        action_ins      = agents[0].action
        df_full         = self.allpopulationinformation.df_full
        X_full          = self.allpopulationinformation.X_full
        
        # Grab global variables
        df_global         = df_full["global"]
        X_global          = X_full["global"]

        # Update insurance premium offer
        if action_ins is not None:

            # Update premium offered post subsidy
            df_global, X_global = self.ApplySubsidy2Premiums(df_global, X_global, EnvTrackers, action_ins, agents)

        # Update people insured
        df_global, X_global, EnvTrackers = self.UpdateInsured(df_global, X_global, EnvTrackers, action_ins)

        # Store back
        df_full["global"]                   = df_global
        X_full["global"]                    = X_global
        
        self.allpopulationinformation.df_full   = df_full
        self.allpopulationinformation.X_full    = X_full
        self.trackingindicators.EnvTrackers     = EnvTrackers

    def update_ins_agent_state(self, t):

        agents          = self.agents
        df_full         = self.allpopulationinformation.df_full
        X_full          = self.allpopulationinformation.X_full
        EnvTrackers     = self.trackingindicators.EnvTrackers

        # Update State Information Insurance
        df_full, X_full = self.State_Update(df_full, X_full, EnvTrackers, "ins", t+1, self.k_cycle["ins"])
        agents[0].state = X_full["ins"]

        self.agents                             = agents
        self.allpopulationinformation.df_full   = df_full
        self.allpopulationinformation.X_full    = X_full

    def apply_hosp_actions(self):

        agents          = self.agents
        EnvTrackers     = self.trackingindicators.EnvTrackers
        action_hosp     = agents[1].action
        df_full         = self.allpopulationinformation.df_full
        X_full          = self.allpopulationinformation.X_full

        # Grab global variables
        df_global         = df_full["global"]
        X_global          = X_full["global"]
        
        # Update hospital queues
        if action_hosp is not None:
            queue_scores = action_hosp
            df_global, X_global, EnvTrackers = self.UpdateHospitalOccupancy(df_global, X_global, queue_scores, EnvTrackers)

        # Store back
        df_full["global"]                   = df_global
        X_full["global"]                    = X_global
        
        self.allpopulationinformation.df_full   = df_full
        self.allpopulationinformation.X_full    = X_full
        self.trackingindicators.EnvTrackers     = EnvTrackers

    def update_hosp_agent_state(self, t):

        agents          = self.agents
        df_full         = self.allpopulationinformation.df_full
        X_full          = self.allpopulationinformation.X_full
        EnvTrackers     = self.trackingindicators.EnvTrackers

        # Update State Information Hospital
        df_full, X_full = self.State_Update(df_full, X_full, EnvTrackers, "hosp", t+1, self.k_cycle["hosp"])
        agents[1].state = X_full["hosp"]

        self.agents                             = agents
        self.allpopulationinformation.df_full   = df_full
        self.allpopulationinformation.X_full    = X_full

    def IllnessUpdates(self, df, X, EnvTrackers, p_in_hosp=0.67):

        N_geog                                      = EnvTrackers["N_geog"]
        feature_start_idxs                          = EnvTrackers['feature_start_idxs']
        feature_extremes                            = EnvTrackers['feature_extremes']
        Geographic_number_healthy_track             = EnvTrackers['Geographic_number_healthy_track']
        Geographic_number_ill_track                 = EnvTrackers['Geographic_number_ill_track']
        Geographic_number_terminated_illness_track  = EnvTrackers['Geographic_number_terminated_illness_track']
        Geographic_number_deceased_track            = EnvTrackers['Geographic_number_deceased_track']
        Geographic_current_available_beds           = EnvTrackers['Geographic_current_available_beds']
        Geographic_number_discharged_track          = EnvTrackers["Geographic_number_discharged_track"]
        params_stillill_outcome                     = EnvTrackers['params_stillill_outcome']
        params_deceased_outcome                     = EnvTrackers['params_deceased_outcome']
        hosp_bill                                   = EnvTrackers['hosp_bill']
        norm_params                                 = EnvTrackers["norm_params"]

        # Parameters for updates
        I_const, I_coeff, I_exp_coeff_x, I_exp_coeff_b, I_exp_coeff_h, I_exp_const = params_stillill_outcome
        D_const, D_coeff, D_exp_coeff_x, D_exp_coeff_b, D_exp_coeff_h, D_exp_const = params_deceased_outcome

        # Note:
        # DECEASED = 0, ILLNESS = 0 (MEANS HEALTHY AND ALIVE)
        # DECEASED = 1, ILLNESS = 0 (MEANS PASSED AWAY)
        # DECEASED = 0, ILLNESS = 1 (MEANS ILL AND ALIVE)

        #---------------Update those with illnesses to see their new state (still ill/cured/deceased)---------------
        # Ill or not
        max_health                                          = feature_extremes['HEALTH'][1]
        idxs_ill                                            = df['ILLNESS']==1
        df_with                                             = df[idxs_ill]
        X_with                                              = X[idxs_ill]
        prob_has_illness                                    = self.prob_health_calc(df_with['ILLTIME'], df_with['WAITBED'], (5-df_with['HEALTH']), I_const, I_coeff, I_exp_coeff_x, I_exp_coeff_b, I_exp_coeff_h, I_exp_const) # Probability of still having the illness given you had it last time step
        one                                                 = np.ones(prob_has_illness.shape[0], dtype=int)
        ill_YN_still                                        = binomial(one, prob_has_illness)               # Yes/No illness assignment
        df_with['ILLNESS']                                  = ill_YN_still                                  # Assign ill/not decisions to people who were previously ill
        X_with[:, feature_start_idxs['ILLNESS'][1]]         = ill_YN_still

        # Of those no longer ill, are they deceased or not?
        idx_no_longer_ill                                   = ill_YN_still==0
        df_no_illness                                       = df_with[idx_no_longer_ill]                    # Get people who don't have the illness anymore
        X_no_illness                                        = X_with[idx_no_longer_ill]
        X_no_illness                                        = X_with[idx_no_longer_ill]
        prob_cured                                          = self.prob_health_calc(df_no_illness['ILLTIME'], df_no_illness['WAITBED'], df_no_illness['HEALTH'], D_const, D_coeff, D_exp_coeff_x, D_exp_coeff_b, D_exp_coeff_h, D_exp_const)  # Probability of passing away given you have illness
        one                                                 = np.ones(prob_cured.shape[0], dtype=int)
        deceased_YN                                         = binomial(one, 1-prob_cured)                   # Yes/No illness assignment
        df_no_illness['DECEASED']                           = deceased_YN                                   # Assign cured/deceased decisions to people who used to have illness and no longer do
        X_no_illness[:,feature_start_idxs['DECEASED'][1]]   = deceased_YN        
        df_no_illness["ILLTIME"]                            = 0                                             # Reset illness times to zero since no longer ill
        X_no_illness[:, feature_start_idxs['ILLTIME'][1]]   = 0
        df_no_illness["NEEDBED"]                            = 0                                             # Reset bed needs to zero since no longer ill
        X_no_illness[:, feature_start_idxs['NEEDBED'][1]]   = 0
        df_no_illness["WAITBED"]                            = 0                                             # Reset time waited for bed to 0 (got better before getting bed. Should we track this differently?)
        X_no_illness[:, feature_start_idxs['WAITBED'][1]]   = 0
        df_with[idx_no_longer_ill]                          = df_no_illness                                 # Assign cured/deceased decisions to people who used to have illness and no longer do
        X_with[idx_no_longer_ill]                           = X_no_illness
        df[idxs_ill]                                        = df_with                                       # Include all updates in main df
        X[idxs_ill]                                         = X_with
    
        #---------------Discharge from hospital those who pass away or are cured---------------
        # Zero out everyone's hospital costs to start time step
        df["HOSPCOST"] = 0
        X[:, feature_start_idxs["HOSPCOST"][1]] = 0

        # Indicator tracking variables
        Discharge_Arr       = np.zeros((N_geog,1))

        # Update information on people in hospital and remove people from hospital who have finshed their stays
        for n_g in range(N_geog):

            # Get features of people in hospital
            idx_in_hosp = np.logical_and(df["INHOSP"], df["REGION"]==n_g).to_numpy().astype(bool)
            df_geog_in_hosp = df[idx_in_hosp]
            X_geog_in_hosp = X[idx_in_hosp]

            # Discharge people from hospital who no longer have illness
            discharge_idx = (df_geog_in_hosp["ILLNESS"] == 0).to_numpy()
            df_geog_in_hosp["INHOSP"].iloc[discharge_idx] = False                       
            X_geog_in_hosp[discharge_idx, feature_start_idxs["INHOSP"][1] ] = 0

            # Assign payment for stay
            df_geog_in_hosp["HOSPCOST"].iloc[discharge_idx] =  hosp_bill                  
            X_geog_in_hosp[discharge_idx, feature_start_idxs["HOSPCOST"][1] ] = hosp_bill/norm_params["const_div"][feature_start_idxs["HOSPCOST"][0]]

            # Update global pool with these feature updates
            df[idx_in_hosp] = df_geog_in_hosp                                           
            X[idx_in_hosp]  = X_geog_in_hosp                                            

            # Update tracking variables
            N_discharged = sum(discharge_idx)
            Geographic_current_available_beds[n_g] += N_discharged            # Update number of available beds
            Discharge_Arr[n_g]       = N_discharged

        # Update tracking variables
        Geographic_number_discharged_track = np.hstack((Geographic_number_discharged_track,Discharge_Arr))

        #---------------Update those without illnesses to see if they contract one---------------    
        # Get those without illnesses entering this module
        idxs_not_ill = ~idxs_ill
        df_without = df[idxs_not_ill]

        # Calculate probability of contracting illness
        prob =  0.4*(1-df_without['HICOV']) + (0.08*df_without['HEALTH']) 
        one = np.ones(prob.shape[0],dtype=int)

        # Yes/No illness assignment
        ill_YN_new =   binomial(one, prob)
        df['ILLNESS'][idxs_not_ill] = ill_YN_new                            # Update df with people having illnesses
        X[idxs_not_ill, feature_start_idxs['ILLNESS'][1]] = ill_YN_new      # Update feature vectors with people having illnesses
        df['NEEDBED'][idxs_not_ill] = ill_YN_new                            # Update df with people needing hospital beds (everyone with illnesses)
        X[idxs_not_ill, feature_start_idxs['NEEDBED'][1]] = ill_YN_new      # Update feature vectors with people needing hospital beds (everyone with illnesses)
    
        # Update tracking indicator variables
        N_healthy             = np.zeros((N_geog,1))
        N_ill                 = np.zeros((N_geog,1))
        N_deceased            = np.zeros((N_geog,1))
        N_terminated_illness  = np.zeros((N_geog,1))
        for n_g in range(N_geog):
            N_healthy[n_g]              = len(ill_YN_new[df["REGION"][idxs_not_ill]==n_g])
            N_ill[n_g]                  = sum(ill_YN_new[df["REGION"][idxs_not_ill]==n_g])
            N_terminated_illness[n_g]   = len(deceased_YN[df_no_illness["REGION"][idxs_ill]==n_g])
            N_deceased[n_g]             = sum(deceased_YN[df_no_illness["REGION"][idxs_ill]==n_g])

        Geographic_number_healthy_track             = np.hstack((Geographic_number_healthy_track, N_healthy))
        Geographic_number_ill_track                 = np.hstack((Geographic_number_ill_track, N_ill))
        Geographic_number_terminated_illness_track  = np.hstack((Geographic_number_terminated_illness_track, N_terminated_illness))
        Geographic_number_deceased_track            = np.hstack((Geographic_number_deceased_track, N_deceased))

        # Add time step to everyone who has illness
        df['ILLTIME'][df['ILLNESS']==True] += 1
        X[:, feature_start_idxs["ILLTIME"][1]] = df["ILLTIME"] / norm_params["const_div"][feature_start_idxs["ILLTIME"][0]]

        EnvTrackers['Geographic_number_healthy_track']              = Geographic_number_healthy_track
        EnvTrackers['Geographic_number_ill_track']                  = Geographic_number_ill_track
        EnvTrackers['Geographic_number_terminated_illness_track']   = Geographic_number_terminated_illness_track
        EnvTrackers['Geographic_number_deceased_track']             = Geographic_number_deceased_track
        EnvTrackers['Geographic_current_available_beds']            = Geographic_current_available_beds
        EnvTrackers['Geographic_number_discharged_track']           = Geographic_number_discharged_track

        return df, X, EnvTrackers

    def prob_health_calc(self, x, b, h, const, coeff, exp_coeff_x, exp_coeff_b, exp_coeff_h, exp_const):

        exp   = exp_coeff_x*x + exp_coeff_b*b + exp_coeff_h*h + exp_const
        if coeff<0:
            coeff = np.abs(coeff)
            probs = const-coeff**(exp)
        else:
            probs = const+coeff**(exp)

        probs = np.clip(probs, 0, 1)
    
        return probs
    
    def UpdateInsuranceProfits(self, df, EnvTrackers):

        Insurance_profits       = EnvTrackers["Insurance_profits_track"]
        InsExpenses             = EnvTrackers["Geographic_payments_to_ins_track"]
        OutofPocketExpensesHosp = EnvTrackers["Geographic_outpocket_hosp_expenses_track"]
        feature_start_idxs      = EnvTrackers['feature_start_idxs']
        N_geog                  = EnvTrackers['N_geog']

        # Get indices of insured people
        idx_insured = df["HICOV"]==1

        # Calculate amonunt paid to insurance
        df_insured = df[idx_insured]
        Insurance_profits = np.append( Insurance_profits, sum(df_insured["HIPFULLCOST"]) - sum(df_insured["HOSPCOST"]) )

        # Calculate out of pocket expenses for hospital visits
        df_uninsured = df[~idx_insured]
        Outpocket_hosp_arr = np.zeros((N_geog,1))
        for n_g in range(N_geog):
            Outpocket_hosp_arr[n_g] = sum(df_insured["HOSPCOST"][df["REGION"]==n_g])
        OutofPocketExpensesHosp = np.hstack((OutofPocketExpensesHosp,Outpocket_hosp_arr))

        # Calculate out of pocket expenses expenses paid to insurance
        Payments_to_ins_arr = np.zeros((N_geog,1))
        for n_g in range(N_geog):
            Payments_to_ins_arr[n_g] = sum(df_insured["HIPCOST"][df["REGION"]==n_g])
        InsExpenses = np.hstack((InsExpenses,Payments_to_ins_arr))

        EnvTrackers["Insurance_profits_track"]                  = Insurance_profits
        EnvTrackers["Geographic_payments_to_ins_track"]         = InsExpenses
        EnvTrackers["Geographic_outpocket_hosp_expenses_track"] = OutofPocketExpensesHosp

        return EnvTrackers


    def UpdateRewardsTrackers(self, EnvTrackers):
        geo_illness             = EnvTrackers['Geographic_number_ill_track']
        geo_deceased            = EnvTrackers['Geographic_number_deceased_track']
        geo_healthy             = EnvTrackers["Geographic_number_healthy_track"]
        geo_terminated_illness  = EnvTrackers["Geographic_number_terminated_illness_track"]
        geo_tot_ins             = EnvTrackers['Geographic_number_insured_track']
        geo_tot                 = EnvTrackers['Geographic_number_total_track']
        ins_profits             = EnvTrackers['Insurance_profits_track']

        # Perform calculations for certain rewards

        # Treat first time step different (nothing to calculate difference of)
        tot_ins_profits_curr    = ins_profits[-1]
        if geo_tot_ins.shape[1]>1:
            geo_tot_healthy_curr            = geo_healthy[:,-1]
            geo_tot_illness_curr            = geo_illness[:,-1]
            geo_tot_terminated_illness_curr = geo_terminated_illness[:,-1]
            geo_tot_deceased_curr           = geo_deceased[:,-1]
            geo_tot_curr                    = geo_tot[:,-1]
            geo_tot_ins_curr                = geo_tot_ins[:,-1]
        else:
            geo_tot_healthy_curr            = geo_healthy
            geo_tot_illness_curr            = geo_illness
            geo_tot_terminated_illness_curr = geo_terminated_illness
            geo_tot_deceased_curr           = geo_deceased
            geo_tot_curr                    = geo_tot
            geo_tot_ins_curr                = geo_tot_ins

        # Gather all rewards
        rewards                 = np.array([tot_ins_profits_curr,
                                            *geo_tot_curr.flatten(),
                                            *geo_tot_deceased_curr.flatten(),
                                            *geo_tot_ins_curr.flatten(),
                                            *geo_tot_illness_curr.flatten(),
                                            *geo_tot_healthy_curr.flatten(),
                                            *geo_tot_terminated_illness_curr.flatten()])
        
        EnvTrackers["rewards"]  = rewards
    
        return EnvTrackers

    def checkFailedEpisode(self, df, X, EnvTrackers):

        ins_profits = EnvTrackers['Insurance_profits_track']
        N_people    = X['global'].shape[0]

        # If no people in system, or insurance makes negative cummulative profits,
        # episode is a fail
        if sum(ins_profits) < 0 or N_people == 0:
            fail = True
        else:
            fail = False

        return fail


    def UpdatePopulationFeatures(self, df, X, EnvTrackers, agents):

        N_geog                      = EnvTrackers["N_geog"]
        updateable_feature          = EnvTrackers["updateable_feature"]
        feature_polarity            = EnvTrackers["feature_polarity"]
        feature_extremes            = EnvTrackers["feature_extremes"]
        feature_increment           = EnvTrackers["feature_increment"]
        public_subsidies            = agents[2].public_subsidy
        prop_max_subsidy            = EnvTrackers["Planner_budget"]/(2*N_geog)        
        deter_const                 = EnvTrackers["deterioration_public"]

        # Update features of people
        for n_g in range(N_geog):

            df_curr_geog = df[df["REGION"]==n_g]
            X_curr_geog = X[df["REGION"]==n_g]
            curr_subsidy = public_subsidies[n_g]

            # Select who gets positive and negative feature updates
            N_samp,_ = X_curr_geog.shape
            prob_calc = 0.2928055160151634+ 0.4*self.sig((curr_subsidy-deter_const-prop_max_subsidy*0.5)*4/(prop_max_subsidy*0.5))
            threshs = [prob_calc, prob_calc+0.2]
            samp = uniform(np.zeros(df_curr_geog.shape[0]),np.ones(df_curr_geog.shape[0]))
            upgrade_samp_idx   = samp<=threshs[0]
            downgrade_samp_idx = samp>threshs[1]
            for n_f in range(len(updateable_feature)):

                curr_feature            = updateable_feature[n_f]

                # We update HEALTH later as a function of these feature updates
                if curr_feature != "HEALTH":
                    curr_feature_extremes       = feature_extremes[curr_feature]
                    curr_feature_polarity       = feature_polarity[curr_feature]
                    df_curr_geog, X_curr_geog   = self.FeaturesUpdate(df_curr_geog, X_curr_geog, EnvTrackers, upgrade_samp_idx,
                                                       downgrade_samp_idx, curr_feature, curr_feature_extremes,
                                                       curr_feature_polarity, feature_increment)

            df[df["REGION"]==n_g] = df_curr_geog
            X[df["REGION"]==n_g] = X_curr_geog

        # Update HEALTH feature
        df, X = self.HealthUpdate(df, X, EnvTrackers)

        return df, X, EnvTrackers

    def HealthUpdate(self, df, X, EnvTrackers):

        # Update the HEALTH feature for everyone in the system who has the illness for the current time step.
        reg                         = EnvTrackers["health_regressor"]
        norm_params                 = EnvTrackers["norm_params"]
        feature_extremes            = EnvTrackers["feature_extremes"]
        feature_start_idxs          = EnvTrackers["feature_start_idxs"]
        min_health                  = feature_extremes['HEALTH'][0]
        max_health                  = feature_extremes['HEALTH'][1]

        # Separate health feature from other variables
        N_features = X.shape[1]
        delete_features = N_features-np.linspace(1,11,11, dtype=int)
        delete_features = np.append(delete_features, feature_start_idxs['HEALTH'][1])
        X_wo_HEALTH = np.delete(X, delete_features, axis=1)

        # Predict person's health given their features
        x_health = reg.predict(X_wo_HEALTH)
        x_health = np.clip(x_health, feature_extremes["HEALTH"][0], feature_extremes["HEALTH"][1])

        # Update population data
        df['HEALTH'] = x_health
        X[:,feature_start_idxs['HEALTH'][1]] = (df['HEALTH'] \
                                            -norm_params["mu_test"][feature_start_idxs["HEALTH"][0]])/norm_params["sig_test"][feature_start_idxs["HEALTH"][0]]

        return df, X

    def continuous_feature_update_func(self, x, plus_minus, min_val, max_val, increment):

        # Increment of improvement
        full_range = (max_val-min_val)
        inc = full_range/increment
        delta = plus_minus*inc
        x_delta = (x + delta)
        out_val = x_delta
        out_val = np.clip(out_val, min_val, max_val)

        return out_val

    def FeaturesUpdate(self, df, X, EnvTrackers, upgrade_samp_idx,
                   downgrade_samp_idx, feature, feature_extremes,
                   feature_polarity, feature_increment):

        norm_params         = EnvTrackers['norm_params']
        feature_names       = EnvTrackers["feature_names"]
        feature_start_idxs  = EnvTrackers["feature_start_idxs"]

        feature_minimum, feature_maximum = feature_extremes
        idx_feature, start_idx,stop_idx = feature_start_idxs[feature]


        # Check if feature is discrete or continuous
        idx_feature = np.argwhere(np.array(feature_names) == feature).item()
        if not isinstance(norm_params['mu_train'][idx_feature], float):
            datatype = 'discrete'
        else:
            datatype = 'continuous'


        # If discrete, apply appropriate updates
        if datatype == 'discrete':
                                                                                    # Incorporate feature_minimum/feature_maximum functionality
            # Update improved samples
            df[feature].iloc[upgrade_samp_idx] = min([df[feature].iloc[upgrade_samp_idx] + feature_polarity, feature_maximum])

            # Update deteriorated samples
            df[feature].iloc[downgrade_samp_idx]  = max([df[feature].iloc[downgrade_samp_idx] - feature_polarity, feature_minimum])
            new_binaries = df[feature].to_numpy()
            X_update = X[:, start_idx:stop_idx+1]
            N_cols = X_update.shape
            vals = np.zeros((1,N_cols))
            X_update[new_binaries==N_cols,:] = vals
            for ii in range(N_cols):
                vals_curr = vals.copy()
                vals_curr[ii] = 1
                X_update[new_binaries==ii,:] = vals_curr
            X[:, start_idx:stop_idx+1] = X_update

        # If continuous, apply appropriate updates    
        else:
        
            # Update improved samples
            x_upgrade = df[feature].iloc[upgrade_samp_idx].to_numpy().astype(float)
            df_new_vals = self.continuous_feature_update_func(x_upgrade, feature_polarity, feature_minimum, feature_maximum, feature_increment)
            df[feature].iloc[upgrade_samp_idx] = df_new_vals

            # Update deteriorated samples
            x_downgrade = df[feature].iloc[downgrade_samp_idx].to_numpy().astype(float)
            df_new_vals = self.continuous_feature_update_func(x_downgrade, -feature_polarity, feature_minimum, feature_maximum, feature_increment)
            df[feature].iloc[downgrade_samp_idx] = df_new_vals

            # Get parameter normalization information
            mu_param_curr = norm_params['mu_test'][idx_feature]
            sig_param_curr = norm_params['sig_test'][idx_feature]
            X[:,start_idx] = ((df[feature].to_numpy())-mu_param_curr)/sig_param_curr
        
        return df, X

    def UpdateHospitalInfrastructure(self, df, X, agents, EnvTrackers):

        base_cost, prop_cost                = EnvTrackers["hospital_bed_cost_params"]
        Geographic_constructing_beds        = EnvTrackers["Geographic_constructing_beds"]
        Geographic_current_available_beds   = EnvTrackers["Geographic_current_available_beds"]
        Geographic_current_total_beds       = EnvTrackers["Geographic_current_total_beds"]
        Geographic_number_total_beds_track  = EnvTrackers["Geographic_number_total_beds_track"]
        Geographic_constructing_time        = EnvTrackers["Geographic_constructing_time"]
        prop_time, base_time                = EnvTrackers["time_coeffs_bed_construct"]
        deterioration_hosp_infra            = EnvTrackers["deterioration_hosp_infra"]
        N_geog                              = EnvTrackers["N_geog"]
        feature_start_idxs                  = EnvTrackers["feature_start_idxs"]
        norm_params                         = EnvTrackers['norm_params']
        hosp_subsidies                      = agents[1].hosp_subsidy
        rollover                            = agents[2].rollover_subsidy

        # Add new projects for building beds into the hospital infrastructure plan
        if hosp_subsidies is not None:

            for n_g in range(N_geog):

                curr_subsidy = hosp_subsidies[n_g]
                N_new_beds = int(np.max([np.floor((curr_subsidy-base_cost)/prop_cost),0]))                  # Number of beds invested in this round
        
                if N_new_beds> 0:
                    rollover += curr_subsidy - (N_new_beds*prop_cost + base_cost)                           # What goes unspent on beds gets rolled over
                    Geographic_constructing_beds[n_g] = np.append(Geographic_constructing_beds[n_g], N_new_beds)
                    time2build = np.round(prop_time*N_new_beds + base_time)
                    Geographic_constructing_time[n_g] = np.append(Geographic_constructing_time[n_g],time2build)
                else:
                    rollover += curr_subsidy

        else:
            rollover = None

        # Update time on all existing infrastructure plans being built
        for n_g in range(N_geog):

            Geographic_constructing_time[n_g] -= 1
            finished_projects = Geographic_constructing_time[n_g]==0
            N_new_beds = sum(Geographic_constructing_beds[n_g][finished_projects])  
            Geographic_constructing_beds[n_g] = np.delete(Geographic_constructing_beds[n_g],finished_projects)  # Remove finished projects from constructing
            Geographic_constructing_time[n_g] = np.delete(Geographic_constructing_time[n_g],finished_projects)
            Geographic_current_total_beds[n_g]     += N_new_beds
            Geographic_current_available_beds[n_g] += N_new_beds

        # Update tracking indicator variables
        TotalBed_Arr = np.zeros((N_geog,1))
        for n_g in range(N_geog):
            TotalBed_Arr[n_g] = Geographic_current_total_beds[n_g]
            df["NGEOBED"][df["REGION"]==n_g] = TotalBed_Arr[n_g].item()
            X[df["REGION"]==n_g, feature_start_idxs["NGEOBED"][1]] = TotalBed_Arr[n_g].item()/norm_params["const_div"][feature_start_idxs["NGEOBED"][0]]
        Geographic_number_total_beds_track = np.hstack((Geographic_number_total_beds_track,TotalBed_Arr))

        # Update EnvTrackers and send back
        EnvTrackers["Geographic_constructing_beds"]         = Geographic_constructing_beds
        EnvTrackers["Geographic_current_available_beds"]    = Geographic_current_available_beds
        EnvTrackers["Geographic_current_total_beds"]        = Geographic_current_total_beds
        EnvTrackers["Geographic_number_total_beds_track"]   = Geographic_number_total_beds_track
        agents[2].rollover_subsidy                          = rollover

        return df, X, EnvTrackers, agents


    def ApplySubsidy2Premiums(self, df, X, EnvTrackers, premiums_offered, agents, rule="Proportional2Income"):

        N_geog              = EnvTrackers["N_geog"]
        feature_start_idxs  = EnvTrackers["feature_start_idxs"]
        norm_params         = EnvTrackers["norm_params"]
        k_cycle_ins         = EnvTrackers["cycle_values"]["ins"]
        ins_geog_subsidies  = agents[0].ins_subsidy

        if rule == "Proportional2Income":

            for n_g in range(N_geog):

                df_curr_geog = df[df["REGION"] == n_g]
                X_curr_geog =  X[df["REGION"] == n_g]
                premiums_offered_geog = premiums_offered[ df["REGION"] == n_g ]

                if ins_geog_subsidies is not None:

                    # Get current geographic location
                    curr_subsidy = ins_geog_subsidies[n_g]/k_cycle_ins    # <----because lump sum amount needs to be divided over each payment in the cycle
                
                    #  Calculate weights for proportion of subsidy that goes to each person and apply subsidy
                    weights = (df_curr_geog['FAMSIZE']/(np.log(df_curr_geog['FTOTVAL'].to_numpy().astype(np.float)+1000)))
                    weights = (weights/sum(weights)).to_numpy()
                    premium_calc = self.adjust_premiums(df_curr_geog, premiums_offered_geog, weights, curr_subsidy)

                    # Update Premiums
                    df_curr_geog['HIPCOST']     = premium_calc
                    df_curr_geog['HIPFULLCOST'] = premiums_offered_geog

                else:
                
                    # Update Premiums
                    df_curr_geog['HIPCOST']     = premiums_offered_geog
                    df_curr_geog['HIPFULLCOST'] = premiums_offered_geog

                X_curr_geog[:, feature_start_idxs['HIPCOST'][1]] = df_curr_geog['HIPCOST']/norm_params['const_div'][feature_start_idxs['HIPCOST'][0]]
                X_curr_geog[:, feature_start_idxs['HIPFULLCOST'][1]] = df_curr_geog['HIPFULLCOST']/norm_params['const_div'][feature_start_idxs['HIPFULLCOST'][0]]

                # Put this information back into the general dataframe
                df[df["REGION"] == n_g]  = df_curr_geog
                X[df["REGION"] == n_g] = X_curr_geog

        return df, X

    def adjust_premiums(self, df, premiums_offered, weights, subsidy):

        N_samp,_ = df.shape

        # Order from poorest to wealthiest when applying subsidies
        idxs_ordered = np.linspace(0,N_samp-1,N_samp)
        premium_calc_arr = np.vstack((idxs_ordered,premiums_offered,weights))
        premium_calc_arr = premium_calc_arr[:,np.argsort(premium_calc_arr[-1,:])[::-1]]
    
        left_over=0

        # Update preimum for each person (one at a time because of poor-over effect
        for nn in range(N_samp):

            # Adjust premium for subsidy (use up all left over subsidy on next poorest person)
            adjusted_premium = premium_calc_arr[1,nn] - premium_calc_arr[2,nn]*subsidy - left_over
            left_over = 0

            # If subsidy greater than premium, treat as left over money to poor over
            # for the next preson's subsidy
            if adjusted_premium<0:
                left_over += -adjusted_premium
                adjusted_premium = 0

            # Apply premium
            premium_calc_arr[1,nn] = adjusted_premium

        # Reorder back to orignal sort
        premium_calc_arr = premium_calc_arr[:,np.argsort(premium_calc_arr[0,:])]
        
        return premium_calc_arr[1,:]

    def UpdateInsured(self, df, X, EnvTrackers, premiums_offered):

        NumberInsured = EnvTrackers["Geographic_number_insured_track"]
        NumberUninsured = EnvTrackers["Geographic_number_uninsured_track"]
        TotalNumber = EnvTrackers["Geographic_number_total_track"]
        N_geog = EnvTrackers["N_geog"]
        feature_start_idxs  = EnvTrackers["feature_start_idxs"]

        if premiums_offered is not None:

            # Get customer decisisons
            decisions = self.CustomerInsuranceDecision(df)

            # Update features of customers
            df['HICOV'] = decisions
            X[:, feature_start_idxs['HICOV'][1]] = decisions

        # Calculate total number of people in this time step in each region
        N_tot_arr = np.zeros((N_geog,1))
        for n_g in range(N_geog):
            N_tot_arr[n_g] = sum(df["REGION"]==n_g)
        TotalNumber = np.hstack((TotalNumber,N_tot_arr))

        # Get indices of insured people
        idx_insured = df["HICOV"]==1
    
        # Calculate number insured
        N_ins_arr = np.zeros((N_geog,1))
        for n_g in range(N_geog):
            N_ins_arr[n_g] = sum(idx_insured[df["REGION"]==n_g])
        NumberInsured = np.hstack((NumberInsured,N_ins_arr))

        # Calculate number insured
        df_uninsured = df[~idx_insured]
        N_unins_arr = np.zeros((N_geog,1))
        for n_g in range(N_geog):
            N_unins_arr[n_g] = sum(~idx_insured[df["REGION"]==n_g])
        NumberUninsured = np.hstack((NumberUninsured,N_unins_arr))

        EnvTrackers["Geographic_number_insured_track"] = NumberInsured
        EnvTrackers["Geographic_number_uninsured_track"] = NumberUninsured
        EnvTrackers["Geographic_number_total_track"] = TotalNumber
        
        return df, X,  EnvTrackers

    def CustomerInsuranceDecision(self, df, const = 300):

        # Propensity to be insured based on simple hueristic
        value = ((df['FTOTVAL']/(df['HIPCOST']*df['FAMSIZE']+1e-9)).to_numpy().astype(float))
        propensity = 1-np.exp(-value/const)
        one = np.ones(len(propensity),dtype=int)
        decisions = binomial(one, propensity)
    
        return decisions

    def UpdateHospitalOccupancy(self, df, X, queue_scores, EnvTrackers):

        hosp_bill                                   = EnvTrackers["hosp_bill"]
        N_geog                                      = EnvTrackers["N_geog"]
        Geographic_current_available_beds           = EnvTrackers["Geographic_current_available_beds"]
        Geographic_total_number_want_bed_track      = EnvTrackers["Geographic_total_number_want_bed_track"]
        Geographic_number_given_bed_track           = EnvTrackers["Geographic_number_given_bed_track"]
        Geographic_wait_times_admitted_track        = EnvTrackers["Geographic_wait_times_admitted_track"]
        feature_start_idxs                          = EnvTrackers["feature_start_idxs"]
        norm_params                                 = EnvTrackers["norm_params"]

        # Indicator tracking variables
        TotalWantingBed_Arr = np.zeros((N_geog,1))
        NumGivenBed_arr     = np.zeros((N_geog,1))
        WaitTimeBed_arr     = np.zeros((N_geog,1))

        # Get people waiting for beds
        idx_waiting_bed = (df["NEEDBED"]==True).to_numpy().astype(bool)
        df_waiting_obs = df[idx_waiting_bed]                                                      
        X_waiting_obs = X[idx_waiting_bed]
    
        # Add people to hospital based on available beds
        for n_g in range(N_geog):

            # Provide beds to those with priority (based on agent rankings)
            N_avail_g = Geographic_current_available_beds[n_g,-1]                                               # Number of beds available in hospital

            # Get people waiting in particular region
            idx_waiting_geog = (df_waiting_obs["REGION"]==n_g).to_numpy().astype(bool)
            N_need_g = sum(idx_waiting_geog)                                                                    # Number of beds needed by population

            # Check if anyone even needs a bed
            if N_need_g>0:
            
                # Find people who will be given beds based on score rankings
                N_given_g = min([N_need_g, N_avail_g])
                df_geog_hosp_obs = df_waiting_obs[idx_waiting_geog]                                             # Get people waiting for beds
                Geographic_current_available_beds[n_g,-1] = N_avail_g - N_given_g
                X_geog_hosp_obs = X_waiting_obs[idx_waiting_geog]
                queue_scores_geog = queue_scores[idx_waiting_geog]
                idx = np.linspace(0,N_need_g-1,N_need_g, dtype=int)
                idx_priority = np.argsort(queue_scores_geog)[::-1][:N_given_g]                                  # Provide priority people with beds (based on agent action)
                idx_non_priority = np.setdiff1d(idx,idx_priority)

                # Update features of those newly added to the hospital
                df_geog_hosp_obs["NEEDBED"].iloc[idx_priority] = False	                                        # Update features of those in the hospital
                df_geog_hosp_obs["INHOSP"].iloc[idx_priority] = True
                X_geog_hosp_obs[idx_priority, feature_start_idxs["NEEDBED"][1]] = 0
                X_geog_hosp_obs[idx_priority, feature_start_idxs["INHOSP"][1]] = 1

                # Updated trackers
                TotalWantingBed_Arr[n_g] = N_need_g
                NumGivenBed_arr[n_g]     = N_given_g
                WaitTimeBed_arr[n_g]     = np.mean(df_geog_hosp_obs["WAITBED"].iloc[idx_priority])

                # Update geographical df
                df_geog_hosp_obs["WAITBED"].iloc[idx_non_priority] += 1	                                        # Increment time waited by people for bed
                X_geog_hosp_obs[idx_non_priority, feature_start_idxs["WAITBED"][1]] = df_geog_hosp_obs["WAITBED"].iloc[idx_non_priority].to_numpy() / norm_params["const_div"][feature_start_idxs["WAITBED"][0]]

                # Incorporate geographical updates of people who were waiting for a bed in df
                df_waiting_obs[idx_waiting_geog]    = df_geog_hosp_obs                                                  
                X_waiting_obs[idx_waiting_geog]     = X_geog_hosp_obs
                df[idx_waiting_bed] = df_waiting_obs	                        
                X[idx_waiting_bed] = X_waiting_obs
            else:
            
                # Updated trackers (nothing to report)
                TotalWantingBed_Arr[n_g] = 0
                NumGivenBed_arr[n_g]     = 0
                WaitTimeBed_arr[n_g]     = 0
                
        # Updated trackers
        Geographic_total_number_want_bed_track = np.hstack((Geographic_total_number_want_bed_track, TotalWantingBed_Arr))
        Geographic_number_given_bed_track      = np.hstack((Geographic_number_given_bed_track, NumGivenBed_arr))
        Geographic_wait_times_admitted_track   = np.hstack((Geographic_wait_times_admitted_track, WaitTimeBed_arr))

        EnvTrackers["Geographic_current_available_beds"]       = Geographic_current_available_beds
        EnvTrackers["Geographic_total_number_want_bed_track"]  = Geographic_total_number_want_bed_track
        EnvTrackers["Geographic_number_given_bed_track"]       = Geographic_number_given_bed_track
        EnvTrackers["Geographic_wait_times_admitted_track"]    = Geographic_wait_times_admitted_track
    
        return df, X, EnvTrackers

    def State_Update(self, df_full, X_full, EnvTrackers, which_agent, time_step, k_cycle):

        feature_names           = EnvTrackers['feature_names']
        feature_names_preproc   = EnvTrackers['feature_names_preproc']
        

        # Insurance state/feature information
        if which_agent == "ins":
            if time_step%k_cycle == 0:
                df_ins                   = df_full["global"]
                X_ins                    = X_full["global"]
                remove_feat_idxs         = np.array([var[:2]=="HI" or var=="PLANBUDGET" for var in feature_names	    ])
                remove_feat_idxs_preproc = np.array([var[:2]=="HI" or var=="PLANBUDGET" for var in feature_names_preproc ])
                feat_remove = list(compress(feature_names, remove_feat_idxs))
                df_ins                   = df_ins.drop(columns=feat_remove)    
                X_ins                    = X_ins[:, ~remove_feat_idxs_preproc]
            else:
                df_ins  = None
                X_ins   = None

            # Store insurance states in global storage
            df_full["ins"]           = df_ins
            X_full["ins"]            = X_ins

            return df_full, X_full

        # Hospital state/feature information
        elif which_agent == "hosp":
            if time_step%k_cycle == 0:
                df_hosp                         = df_full["global"]
                X_hosp                          = X_full["global"]
                idx_need_bed                    = df_hosp['NEEDBED'] == True
                df_hosp                         = df_hosp[idx_need_bed]
                X_hosp                          = X_hosp[idx_need_bed]
                remove_feat_idxs                = np.array([var=="PLANBUDGET" for var in feature_names         ])
                remove_feat_idxs_preproc        = np.array([var=="PLANBUDGET" for var in feature_names_preproc ])
                feat_remove                     = list(compress(feature_names, remove_feat_idxs))
                df_hosp                         = df_hosp.drop(columns=feat_remove)
                X_hosp                          = X_hosp[:, ~remove_feat_idxs_preproc]
            
            else:
                df_hosp = None
                X_hosp  = None

            # Store hospital states in global storage
            df_full["hosp"]                  = df_hosp
            X_full["hosp"]                   = X_hosp

            return df_full, X_full

        # Omni. planner state/feature information
        elif which_agent == "planner":
            if time_step%k_cycle == 0:
                df_planner              = df_full["global"]
                X_planner               = X_full["global"]
            else:
                df_planner  = None
                X_planner   = None

            # Store planner states in global storage
            df_full["planner"]      = df_planner
            X_full["planner"]       = X_planner

            return df_full, X_full
