import numpy as np
from multiagent.core_education import World, AgentUNIVAdm, AgentUNIVBudget, AgentEmployer, AgentPlanner, AllPopulationInformation, TrackingIndicators
import os, sys
import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from itertools import compress
import pdb

sys.path.insert(0, '../Longterm/')

''' This script constructs the education environment class used to sample observations from actions made by agents'''
class Environment(object):

    def __init__(self, policyTypes, which_agents, k_cycle, episode_length):
        
        self.world = self.make_world(policyTypes, which_agents, k_cycle, episode_length)
    
    def make_world(self, policyTypes, which_agents, k_cycle, episode_length):

        world = World()
        # Set any world properties first
        world.agents = []

        world.agents.append(AgentUNIVAdm())
        world.agents[0].policyType = policyTypes[0]
        if which_agents[0]:
            world.agents[0].agent_flag = True
        else:
            world.agents[0].agent_flag = False

        world.agents.append(AgentUNIVBudget())
        world.agents[1].policyType = policyTypes[1]
        if which_agents[1]:
            world.agents[1].agent_flag = True            
        else:
            world.agents[1].agent_flag = False            

        world.agents.append(AgentEmployer())
        world.agents[2].policyType = policyTypes[2]
        if which_agents[2]:
            world.agents[2].agent_flag = True
        else:
            world.agents[2].agent_flag = False

        world.agents.append(AgentPlanner())
        world.agents[3].policyType = policyTypes[3]
        if which_agents[3]:
            world.agents[3].agent_flag = True
        else:
            world.agents[3].agent_flag = False

        world.allpopulationinformation  = AllPopulationInformation()
        world.trackingindicators        = TrackingIndicators()

        training_path = "./stored_rewards_data/"
        if not os.path.exists(training_path):
                os.mkdir(training_path)

        world.k_cycle           = k_cycle                                       # Cycle/frequency of each action (e.g. action taken every time step, every other time step, ...)
        world.save_dta_flag     = False
        world.N_geog            = 9	                                        # Number of geographic regions
        world.feature_increment = 10                                            # How much of an improvement can we make in a feature at any given point in time
        world.agent_config      = "single"                                      # Specifies whether this agent is sub-centralized or sub-decentralized.        TODO: Start with centralized, but later generalize to decentralized
        world.store_path        = "env_dta/"                                    # Path to store results
        world.tuition           = 5000
        world.episode_length    = 100                                               # Number of time steps of entire simulation
        world.sample_sz         = 500
        
        # Import data and models that are part of the environment
        world = self.loadDtaModels(world)
        
        return world

    def loadDtaModels(self, world):

        # Import needed world variables
        k_cycle             = world.k_cycle
        tuition             = world.tuition
        feature_increment   = world.feature_increment
        save_dta_flag       = world.save_dta_flag
        sample_sz           = world.sample_sz

        # Read in data from dataset
        print("Loading dataset...")
        if save_dta_flag:                                                                       # INCORPORATE feature_extremes VARIABLE
            df_train, df_test, X_train, X_test, feature_names, feature_names_preproc, norm_params, feature_start_idxs, feature_extremes, feature_polarity = load_education_data() # For training loan approval model
            reg = train_gpa_regressor(df_train, X_train, norm_params, feature_start_idxs)
#            pickle.dump(reg, open('./env_dta/uggpa_regressor.pkl', 'wb'))
            pickle.dump(reg, open('./env_dta/uggpa_regressor_11082024.pkl', 'wb'))
            np.savetxt('./env_dta/X_train.txt', X_train)
            df_train.to_pickle('./env_dta/df_train.pkl')
            np.savetxt('./env_dta/X_test.txt', X_test)
            df_test.to_pickle('./env_dta/df_test.pkl')
            np.save('./env_dta/feature_names.npy', feature_names)
            np.save('./env_dta/feature_names_preproc.npy', feature_names_preproc)
            np.save('./env_dta/norm_params.npy', norm_params)
            np.save('./env_dta/feature_start_idxs.npy', feature_start_idxs)
            np.save('./env_dta/feature_extremes.npy',feature_extremes)
            np.save('./env_dta/feature_polarity.npy',feature_polarity)
    
        else:
            reg                     = pickle.load(open('./env_dta/uggpa_regressor_11082024.pkl', 'rb'))
            X_train                 = np.loadtxt('./env_dta/X_train.txt')
            df_train                = pd.read_pickle('./env_dta/df_train.pkl')
            X_test                  = np.loadtxt('./env_dta/X_test.txt')
            df_test                 = pd.read_pickle('./env_dta/df_test.pkl')
            feature_names           = np.load('./env_dta/feature_names.npy').tolist()
            feature_names_preproc   = np.load('./env_dta/feature_names_preproc.npy').tolist()
            norm_params             = np.load('./env_dta/norm_params.npy', allow_pickle=True).item()
            feature_start_idxs      = np.load('./env_dta/feature_start_idxs.npy', allow_pickle=True).item()
            feature_extremes        = np.load('./env_dta/feature_extremes.npy',allow_pickle=True).item()
            feature_polarity        = np.load('./env_dta/feature_polarity.npy',allow_pickle=True).item()

        # Oversample population
        df_test, X_test = self.expand_dta_set(df_test, X_test, ratio=1.67)

        # Import model information to the world
        world.feature_start_idxs    = feature_start_idxs
        world.feature_extremes      = feature_extremes
        world.norm_params           = norm_params
        world.feature_polarity      = feature_polarity
        world.feature_names         = feature_names
        world.feature_names_preproc = feature_names_preproc

        # Final parameter initialization
        scale_const       = {"univ_adm":None, "univ_budget":None, "emp":209515.0945172529, "planner":None}
        world.init_params =   {"tuition"                        :tuition,
                                "k_cycle"                       :k_cycle,
                                "sample_sz"                     :sample_sz,
                                "Planner_budget"                :2.5e7,
                                "uggpa_regressor"               :reg,
                                "updateable_feature"            :list(feature_polarity.keys()),
                                "const_div_INUNIV"              :3000,
                                "const_div_CURRENTGPA"          :4,
                                "const_div_INWORKF"             :3000,
                                "const_div_PLANBUDGET"          :2.5e7,
                                "const_div_UNIVBUDGET"          :2.5e7,
                                "const_div_ANNUALTUIT"          :tuition,
                                "const_div_N_UNIV_UNITS"        :50,
                                "const_div_N_FACULTY"           :200,
                                "const_div_N_STUDENTS_CURR"     :1000,
                                "const_div_TIMEINUNIV"          :6,
                                "const_div_TIMEINWORKF"         :25,
                                "const_div_TIMEINENV"           :40,   
                                "const_div_DIVINVEST"           :2.5e7,
                                "const_div_AVE_SALARY"          :scale_const,
                                "const_div_AGE"                 :40,
                                "feature_start_idxs"            :feature_start_idxs,
                                "feature_increment"             :feature_increment,
                                "feature_extremes"              :feature_extremes,
                                "norm_params"                   :norm_params,
                                "feature_names"                 :feature_names,
                                "feature_names_preproc"         :feature_names_preproc,
                                "feature_polarity"              :feature_polarity,
                                "deterioration_public"          :0,
                                "deterioration_univ_infra"      :2e6,
                                "university_infra_cost_params"  :[1e6, 1e6],
                                "time_coeffs_infra_construct"   :[0.05, 4],
                                "uggpa_regressor"               :reg,
                                "UNIV_budget_rollover"          :2e6,
                                "c_faculty"                     :5,
                                "c_student"                     :5*15, # Should be like c_faculty*15 or something
                                "c_emp_div_incent"              :5,
                                "cost_program"                  :75000,
                                "cost_faculty"                  :75000,        # <-----change during tuning!!
                                "current_total_units"           :10,
                                "current_college_student"       :0,
                                "tuition"                       :tuition,
                                "max_time_in_env"               :40,
                                "scale_const"                   :scale_const,
                                "profitability_weights"         :[10.5/4, 10.5/4, 3],
                                "max_profitability"             :300000,
                                 }

        # Augment data with tracker features (e.g. who is in the university vs workforce, how long they have been in, what degree pursing, etc.)
        X_pop, df_pop, feature_names, feature_names_preproc, norm_params, feature_start_idxs = self.augment_features(X_test,
                                                                                                        df_test,
                                                                                                        feature_names,
                                                                                                        feature_names_preproc,
                                                                                                        norm_params,
                                                                                                        feature_start_idxs,
                                                                                                        world.init_params)
        X_pop, df_pop = self.add_GPA_bias(X_pop, df_pop, reg, feature_start_idxs, feature_extremes)


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

    def add_GPA_bias(self, X, df, reg, feature_start_idxs, feature_extremes):	

        bias_n_g = np.array([4, 0])
        df_bias = df.copy().drop(df.index)
        X_bias = np.zeros((0,X.shape[1]))
        N_features = X.shape[1]
        delete_features = N_features-np.linspace(1,17,17, dtype=int)
        delete_features = np.append(delete_features, feature_start_idxs['UGGPA'][1])

        for n_g in range(2):

            X_geog  = X[df["MINRTY"]==n_g]
            df_geog = df[df["MINRTY"]==n_g]
            N_choice = X_geog.shape[0]
            idx_choices = np.linspace(0, N_choice-1, N_choice, dtype=int)

            X_wo_GPA = np.delete(X_geog, delete_features, axis=1)
            x_health = reg.predict(X_wo_GPA)
            x_health = np.clip(x_health, feature_extremes["UGGPA"][0], feature_extremes["UGGPA"][1])
            dist = (4.1-abs(x_health-bias_n_g[n_g]))**4
            dist = dist/np.sum(dist)
            sample_idxs = np.random.choice(idx_choices, N_choice, replace=True, p= dist)
      
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

    # Do this before each episode starts
    def reset_world(self):
        
        # Set up trackers for environment and learning models
        self.world.trackingindicators.EnvTrackers = self.constructEnvTrackers()

        # Initialize all State/Feature information
        self.init_agent_features()

        # Get first observation to return
        obs = [self.world.agents[0].state, self.world.agents[1].state,
               self.world.agents[2].state, self.world.agents[3].state]

        return obs


    def step(self, iter_num):

        return self.world.step(iter_num)

    def expand_dta_set(self, df, X, ratio=2):

        N               = df.shape[0]
        N_oversample    = np.floor(ratio*N).astype(int)
        sample_idxs     = np.linspace(0,N-1,N)
        oversample_idx  = np.random.choice(sample_idxs, N_oversample, replace=True).astype(int)
        df_new          = df.iloc[oversample_idx]
        X_new           = X[oversample_idx]
        df_new.reset_index()

        return df_new, X_new

    def train_gpa_regressor(self, df, X, norm_params, feature_start_idxs):

        # Train regressor to predict a person's health given their other features
        pdb.set_trace()
        y_train = df['UGGPA'].to_numpy()
        X_train = np.delete(X, feature_start_idxs['UGGPA'][1], axis=1)
        reg = Ridge(alpha=2).fit(X_train, y_train)
        pdb.set_trace()
        return reg

    def augment_features(self, X_pop, df_pop, feature_names, feature_names_preproc, norm_params, feature_start_idxs, init_params):   

        N = df_pop.shape[0]
        # Augment feature_start_idxs
        df_sz = df_pop.shape[1]
        X_sz = X_pop.shape[1]
        feature_start_idxs["INUNIV"]            = (df_sz,    X_sz,    X_sz+1)
        feature_start_idxs["INWORKF"]           = (df_sz+1,  X_sz+1,  X_sz+2)
        feature_start_idxs["INENV"]             = (df_sz+2,  X_sz+2,  X_sz+3)
        feature_start_idxs["INMINTYPGRM"]       = (df_sz+3,  X_sz+3,  X_sz+4)
        feature_start_idxs["CURRENTGPA"]        = (df_sz+4,  X_sz+4,  X_sz+5)
        feature_start_idxs["PLANBUDGET"]        = (df_sz+5,  X_sz+5,  X_sz+6)
        feature_start_idxs["UNIVBUDGET"]        = (df_sz+6,  X_sz+6,  X_sz+7)
        feature_start_idxs["ANNUALTUIT"]        = (df_sz+7,  X_sz+7,  X_sz+8)
        feature_start_idxs["N_UNIV_UNITS"]      = (df_sz+8,  X_sz+8,  X_sz+9)
        feature_start_idxs["N_FACULTY"]         = (df_sz+9,  X_sz+9,  X_sz+10)
        feature_start_idxs["N_STUDENTS_CURR"]   = (df_sz+10, X_sz+10, X_sz+11)
        feature_start_idxs["TIMEINUNIV"]        = (df_sz+11, X_sz+11, X_sz+12)
        feature_start_idxs["TIMEINWORKF"]       = (df_sz+12, X_sz+12, X_sz+13)
        feature_start_idxs["TIMEINENV"]         = (df_sz+13, X_sz+13, X_sz+14)
        feature_start_idxs["DIVINVEST"]         = (df_sz+14, X_sz+14, X_sz+15)
        feature_start_idxs["AGE"]               = (df_sz+15, X_sz+15, X_sz+16)
        feature_start_idxs["AVE_SALARY"]        = (df_sz+16, X_sz+16, X_sz+17)

        # Update feature_names
        feature_names.append('INUNIV')
        feature_names.append('INWORKF')
        feature_names.append('INENV')
        feature_names.append('INMINTYPGRM')
        feature_names.append('CURRENTGPA')
        feature_names.append('PLANBUDGET')
        feature_names.append('UNIVBUDGET')
        feature_names.append('ANNUALTUIT')
        feature_names.append('N_UNIV_UNITS')
        feature_names.append('N_FACULTY')
        feature_names.append('N_STUDENTS_CURR')
        feature_names.append('TIMEINUNIV')
        feature_names.append('TIMEINWORKF')
        feature_names.append('TIMEINENV')
        feature_names.append('DIVINVEST')
        feature_names.append('AGE')
        feature_names.append('AVE_SALARY')

        # Update feature_names_preproc
        feature_names_preproc.append('INUNIV')
        feature_names_preproc.append('INWORKF')
        feature_names_preproc.append('INENV')
        feature_names_preproc.append('INMINTYPGRM')
        feature_names_preproc.append('CURRENTGPA')
        feature_names_preproc.append('PLANBUDGET')
        feature_names_preproc.append('UNIVBUDGET')
        feature_names_preproc.append('ANNUALTUIT')
        feature_names_preproc.append('N_UNIV_UNITS')
        feature_names_preproc.append('N_FACULTY')
        feature_names_preproc.append('N_STUDENTS_CURR')
        feature_names_preproc.append('TIMEINUNIV')
        feature_names_preproc.append('TIMEINWORKF')
        feature_names_preproc.append('TIMEINENV')
        feature_names_preproc.append('DIVINVEST')
        feature_names_preproc.append('AGE')
        feature_names_preproc.append('AVE_SALARY')

        # Update normalization parameters
        const_div = [[]]*len(norm_params['mu_test'])
        const_div.append([])                                            # Binary for INUNIV
        const_div.append([])                                            # Binary for INWORKF
        const_div.append([])                                            # Binary for INENV
        const_div.append([])                                            # Binary for INMINTYPGRM
        const_div.append(init_params["const_div_CURRENTGPA"])           # Contiuous for CURRENTGPA.      Approx guess of max size of current GPA (this past year, not cummulative) (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_PLANBUDGET"])           # Contiuous for PLANBUDGET.      Approx guess of max size of planner budget (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_UNIVBUDGET"])           # Contiuous for UNIVBUDGET.      Approx guess of max size of university budget (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_ANNUALTUIT"])           # Contiuous for ANNUALTUIT.      Approx guess of max size of annual tuition (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_N_UNIV_UNITS"])         # Contiuous for N_UNIV_UNITS.    Approx guess of max size of number of infrastructure units available at university, which affects number of faculty that can be hired and students that can be accepted (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_N_FACULTY"])            # Contiuous for N_FACULTY.       Approx guess of max size of number of infrastructure units available at university, which affects number of faculty that can be hired and students that can be accepted (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_N_STUDENTS_CURR"])      # Contiuous for N_STUDENTS_CURR. Approx guess of max size of number of total students at university at any given point in time
        const_div.append(init_params["const_div_TIMEINUNIV"])           # Contiuous for TIMEINUNIV.      Approx guess of max time in university (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_TIMEINWORKF"])          # Contiuous for TIMEINWORKF.     Approx guess of max time in workforce (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_TIMEINENV"])            # Contiuous for TIMEINENV.       Approx guess of max time for a sample running through the environment (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_DIVINVEST"])            # Contiuous for DIVINVEST.       Approx guess of max time for a sample running through the environment (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_AGE"])                  # Contiuous for AGE.             Approx guess of max time for a sample running through the environment (division by this should hopefully mean max time is [0,1])
        const_div.append(init_params["const_div_AVE_SALARY"])           # Contiuous for AVE_SALARY.      Approx guess of max time for a sample running through the environment (division by this should hopefully mean max time is [0,1])
        extension = [[]]*(len(const_div)-len(norm_params['mu_test']))
        norm_params['mu_train'].extend(extension)
        norm_params['mu_test'].extend(extension)
        norm_params['sig_train'].extend(extension)
        norm_params['sig_test'].extend(extension)
        norm_params['const_div'] = const_div

        # Augment df
        df_pop['INUNIV']            = np.zeros(N,dtype=int)
        df_pop['INWORKF']           = np.zeros(N,dtype=int)
        df_pop['INENV']             = np.zeros(N,dtype=int)
        df_pop['INMINTYPGRM']       = np.zeros(N,dtype=int)
        df_pop['CURRENTGPA']        = np.zeros(N)
        df_pop['PLANBUDGET']        = np.ones(N)*init_params['const_div_PLANBUDGET']
        df_pop['UNIVBUDGET']        = np.zeros(N)*init_params['UNIV_budget_rollover']
        df_pop['ANNUALTUIT']        = np.zeros(N)
        df_pop['N_UNIV_UNITS']      = np.ones(N)*init_params['current_total_units']
        df_pop['N_FACULTY']         = np.ones(N)*init_params['current_total_units']*init_params['c_faculty']
        df_pop['N_STUDENTS_CURR']   = np.zeros(N)
        df_pop['TIMEINUNIV']        = np.zeros(N)
        df_pop['TIMEINWORKF']       = np.zeros(N)
        df_pop['TIMEINENV']         = np.zeros(N)
        df_pop['DIVINVEST']         = np.zeros(N)
        df_pop['AGE']               = np.zeros(N)
        df_pop['AVE_SALARY']        = np.zeros(N)
        df_pop                      = df_pop.astype('object')
        df_pop                      = df_pop.reset_index(drop=True)

        # Augment X
        zero  = np.zeros((N,1))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        aux = np.expand_dims(df_pop['PLANBUDGET']/norm_params["const_div"][feature_start_idxs["PLANBUDGET"][0]],1).astype(float)
        X_pop = np.hstack((X_pop, aux))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        aux = np.expand_dims(df_pop['N_UNIV_UNITS']/norm_params["const_div"][feature_start_idxs["N_UNIV_UNITS"][0]],1).astype(float)
        X_pop = np.hstack((X_pop, aux))
        aux = np.expand_dims(df_pop['N_FACULTY']/norm_params["const_div"][feature_start_idxs["N_FACULTY"][0]],1).astype(float)
        X_pop = np.hstack((X_pop, aux))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))
        X_pop = np.hstack((X_pop, zero))

        return X_pop, df_pop, feature_names, feature_names_preproc, norm_params, feature_start_idxs


    def constructEnvTrackers(self):

        init_params     = self.world.init_params
        N_geog          = self.world.N_geog

        EnvTrackers = {
            "N_geog"                            :N_geog,
            "actions"                           :{"univ_adm":None, "univ_budget":None, "univ_budget":None, "planner":None},
            "each_investment"                   :{"tertiary":None, "univ_budget":None, "emp":None},
            "k_cycle"                           :init_params["k_cycle"],
            "sample_sz"                         :init_params["sample_sz"],
            "Planner_budget"                    :init_params["Planner_budget"],
            "UNIV_budget_rollover"              :init_params["UNIV_budget_rollover"],
            "UNIV_fail_flag"                    :False,
            "EMP_fail_flag"                     :False,
            "uggpa_regressor"                   :init_params["uggpa_regressor"],
            "updateable_feature"                :init_params["updateable_feature"],
            "feature_start_idxs"                :init_params["feature_start_idxs"],
            "feature_increment"                 :init_params["feature_increment"],
            "feature_extremes"                  :init_params["feature_extremes"],
            "norm_params"                       :init_params["norm_params"],
            "feature_names"                     :init_params["feature_names"].copy(),
            "feature_names_preproc"             :init_params["feature_names_preproc"].copy(),
            "feature_polarity"                  :init_params["feature_polarity"],
            "university_infra_cost_params"      :init_params["university_infra_cost_params"],
            "time_coeffs_infra_construct"       :init_params["time_coeffs_infra_construct"],
            "deterioration_public"              :init_params["deterioration_public"],
            "deterioration_univ_infra"          :init_params["deterioration_univ_infra"],
            "c_faculty"                         :init_params["c_faculty"],
            "c_student"                         :init_params["c_student"], # Should be like c_faculty*15 or something
            "c_emp_div_incent"                  :init_params["c_emp_div_incent"],
            "cost_program"                      :init_params["cost_program"],
            "cost_faculty"                      :init_params["cost_faculty"],
            "current_total_units"               :init_params["current_total_units"],
            "current_college_student"           :init_params["current_college_student"],
            "tuition"                           :init_params["tuition"],
            "max_time_in_env"                   :init_params["max_time_in_env"],
            "profitability_weights"             :init_params["profitability_weights"],
            "max_profitability"                 :init_params["max_profitability"],
            "constructing_units"                :np.array([]),
            "constructing_time"                 :np.array([]),
            "emp_profits_track"                 :np.array([]),
            "number_total_units_track"          :np.array([]),
            "number_total_faculty_track"        :np.array([]),
            "Demographic_applied_track"         :np.zeros((2,0)),          #
            "Demographic_admitted_track"        :np.zeros((2,0)),          #
            "Demographic_went_undergrad_track"  :np.zeros((2,0)),          #
            "Demographic_went_masters_track"    :np.zeros((2,0)),          #
            "Demographic_went_doctorate_track"  :np.zeros((2,0)),          #
            "Demographic_graduated_undergrad_track"   :np.zeros((2,0)),    #
            "Demographic_graduated_masters_track"     :np.zeros((2,0)),    #
            "Demographic_graduated_doctorate_track"   :np.zeros((2,0)),    #
            "Demographic_in_workforce_track"    :np.zeros((2,0)),          #
            "Demographic_salary_track"          :np.zeros((2,0)),          #
            "Demographic_in_univ_track"         :np.zeros((2,0)),          #
            "Demographic_total_sum_gpa_track"   :np.zeros((2,0)),          #
            }

        return EnvTrackers 

    def init_agent_features(self):

        # Import needed information from the world
        X_pop          = self.world.allpopulationinformation.X_pop.copy()
        df_pop         = self.world.allpopulationinformation.df_pop.copy()
        EnvTrackers    = self.world.trackingindicators.EnvTrackers

        # Sample people from the population
        X_pop, df_pop = self.sample_from_population(X_pop, df_pop, EnvTrackers, is_init=True)

        # High school graduates state/feature information
        X_univ_adm, df_univ_adm = self.Global2AgentFeatures(X_pop, df_pop, EnvTrackers, "univ_adm", version="agent")

        # In university
        X_univ_budget, df_univ_budget = self.Global2AgentFeatures(X_pop, df_pop, EnvTrackers, "univ_budget", version="agent")

        # Employer state/feature information
        X_emp, df_emp = self.Global2AgentFeatures(X_pop, df_pop, EnvTrackers, "emp", version="agent")

        # Omni. planner state/feature information
        df_planner  =  df_pop
        X_planner   =  X_pop

        X_full = {"global":X_pop, "univ_adm":X_univ_adm, "univ_budget":X_univ_budget, "emp":X_emp, "planner":X_planner}
        df_full = {"global":df_pop, "univ_adm":df_univ_adm, "univ_budget":df_univ_budget, "emp":df_emp, "planner":df_planner}

        self.world.allpopulationinformation.X_full  = X_full
        self.world.allpopulationinformation.df_full = df_full
        self.world.agents[0].state                  = X_univ_adm
        self.world.agents[1].state                  = X_univ_budget
        self.world.agents[2].state                  = X_emp
        self.world.agents[3].state                  = X_planner
        self.world.trackingindicators.EnvTrackers   = EnvTrackers

    def sample_from_population(self, X, df, EnvTrackers, is_init=False):

        if is_init:
            sample_sz = 0
        else:
            sample_sz = EnvTrackers['sample_sz']

        feature_start_idxs = EnvTrackers['feature_start_idxs']

        # Only sample people from population not currently in the environment
        idxs_sampleable = df['INENV'] == 0
        df_sampleable   = df[idxs_sampleable]
        X_sampleable    = X[idxs_sampleable]

        # Sample people to be considered to apply for university or enter the workforce from
        # those not currently inside the environment being processed.
        N_pop = sum(idxs_sampleable)
        idxs = np.linspace(0,N_pop-1, N_pop)
        idxs_sampled = np.random.choice(idxs, sample_sz, replace=False).astype(int)

        # Update variables to indicate who is not being processed
        df_sampleable['INENV'].iloc[idxs_sampled] = 1
        X_sampleable[idxs_sampled, feature_start_idxs['INENV'][1]]  = 1

        # Incorporate updates in global variables
        df[idxs_sampleable] = df_sampleable
        X[idxs_sampleable]  = X_sampleable

        return X, df

    def Global2AgentFeatures(self, X_global, df_global, EnvTrackers, which_agent, version="full", return_idx=False):

        feature_names           =   EnvTrackers["feature_names"]
        feature_names_preproc   =   EnvTrackers["feature_names_preproc"]
        norm_params             =   EnvTrackers["norm_params"]
        feature_start_idxs      =   EnvTrackers["feature_start_idxs"]

        if which_agent=="tertiary":

            idx_tertiary            = ~(df_global['INENV']==1)
            df_tertiary             = df_global[idx_tertiary]
            X_tertiary              = X_global[idx_tertiary]

            if version != "full":
                remove_feat_idxs                = np.array([var=="UGGPA" or var=="AVE_SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="N_UNIV_UNITS" or var=="N_FACULTY" or var=="N_STUDENTS_CURR" or var=="AGE" or var=="ANNUALTUIT" or var=="PLANBUDGET" or var=="UNIVBUDGET" or var=="CURRENTGPA" for var in feature_names	    ])
                remove_feat_idxs_preproc        = np.array([var=="UGGPA" or var=="AVE_SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="N_UNIV_UNITS" or var=="N_FACULTY" or var=="N_STUDENTS_CURR" or var=="AGE" or var=="ANNUALTUIT" or var=="PLANBUDGET" or var=="UNIVBUDGET" or var=="CURRENTGPA" for var in feature_names_preproc ])
                feat_remove                     = list(compress(feature_names, remove_feat_idxs))
                df_tertiary                     = df_tertiary.drop(columns=feat_remove)    
                X_tertiary                      = X_tertiary[:, ~remove_feat_idxs_preproc]

            if return_idx == True:
                return X_tertiary, df_tertiary, idx_tertiary
            else:
                return X_tertiary, df_tertiary

        # High school graduates state/feature information
        if which_agent == "univ_adm":
            
            idx_univ_adm                        = np.logical_and(df_global['TIMEINENV'] == 0, df_global['INENV']==1)
            df_univ_adm                         = df_global[idx_univ_adm]
            X_univ_adm                          = X_global[idx_univ_adm]

            if version != "full":
                remove_feat_idxs                = np.array([var=="AVE_SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="CURRENTGPA" or var=="AGE" or var=="PLANBUDGET" for var in feature_names         ])
                remove_feat_idxs_preproc        = np.array([var=="AVE_SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="CURRENTGPA" or var=="AGE" or var=="PLANBUDGET" for var in feature_names_preproc ])
                feat_remove                     = list(compress(feature_names, remove_feat_idxs))
                df_univ_adm                     = df_univ_adm.drop(columns=feat_remove)    
                X_univ_adm                      = X_univ_adm[:, ~remove_feat_idxs_preproc]

            if return_idx == True:
                return X_univ_adm, df_univ_adm, idx_univ_adm
            else:
                return X_univ_adm, df_univ_adm

        # In university
        elif which_agent == "univ_budget":

            idx_univ_budget                     = np.logical_and(df_global['TIMEINUNIV'] == 0, df_global['INUNIV']==1)
            df_univ_budget                      = df_global[idx_univ_budget]
            X_univ_budget                       = X_global[idx_univ_budget]
            
            if version != "full":
                remove_feat_idxs                = np.array([var=="AVE_SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="CURRENTGPA" or var=="AGE" or var=="PLANBUDGET" for var in feature_names         ])
                remove_feat_idxs_preproc        = np.array([var=="AVE_SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="CURRENTGPA" or var=="AGE" or var=="PLANBUDGET" for var in feature_names_preproc ])
                feat_remove                     = list(compress(feature_names, remove_feat_idxs))
                df_univ_budget                  = df_univ_budget.drop(columns=feat_remove)
                X_univ_budget                   = X_univ_budget[:, ~remove_feat_idxs_preproc]

            if return_idx == True:
                return X_univ_budget, df_univ_budget, idx_univ_budget
            else:
                return X_univ_budget, df_univ_budget
            
        # Employer state/feature information
        elif which_agent == "emp":
            idx_in_workforce                    = df_global['INWORKF'] == 1
            df_emp                              = df_global[idx_in_workforce]
            X_emp                               = X_global[idx_in_workforce]

            if version != "full":
                remove_feat_idxs                = np.array([var=="AVE_SALARY" or var=="SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="N_UNIV_UNITS" or var=="N_FACULTY" or var=="N_STUDENTS_CURR" or var=="ANNUALTUIT" or var=="PLANBUDGET" or var=="UNIVBUDGET" or var=="CURRENTGPA" for var in feature_names         ])
                remove_feat_idxs_preproc        = np.array([var=="AVE_SALARY" or var=="SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="N_UNIV_UNITS" or var=="N_FACULTY" or var=="N_STUDENTS_CURR" or var=="ANNUALTUIT" or var=="PLANBUDGET" or var=="UNIVBUDGET" or var=="CURRENTGPA" for var in feature_names_preproc ])
                feat_remove                     = list(compress(feature_names, remove_feat_idxs))
                df_emp                          = df_emp.drop(columns=feat_remove)
                X_emp                           = X_emp[:, ~remove_feat_idxs_preproc]

            if return_idx == True:
                return X_emp, df_emp, idx_in_workforce
            else:
                return X_emp, df_emp    

    def constructLossTrackers(self):

        LossTrackers = {
        "ins_rewards_arr":      np.array([]),
        "hosp_rewards_arr":     np.array([]),
        "planner_rewards_arr":  np.array([]),
        "mortRts_geoEqty_arr":  np.array([]),
        "insRts_geoEqty_arr":   np.array([])
        }

        return LossTrackers

























            
