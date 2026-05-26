import numpy as np
import pandas as pd
import random
from numpy.random import  binomial, uniform
import pdb
from itertools import compress

class AllPopulationInformation(object):
    def __init__(self): # FILL IN INITIALIZATIONS
        
        # Population applicant feature information
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
    def __init__(self): # FILL IN INITIALIZATIONS

        # System indicators to monitor every round
        self.EnvTrackers = None  # All parameters and things to be tracked inside the environment

# properties of agent entities
class AgentUNIVAdm(object):
    def __init__(self):

        # agent action structure
        self.policyType = None
                        
        # state
        self.state = None
        
        # action
        self.action = None

        # Flags where agent is an RL model (True) or part of the environment (False)
        self.agent_flag = None
        
# properties of agent entities
class AgentUNIVBudget(object):
    def __init__(self):

        # agent action structure
        self.policyType = None
        
        # state
        self.state = None
        
        # action
        self.action = None

        # Flags where agent is an RL model (True) or part of the environment (False)
        self.agent_flag = None

# properties of agent entities
class AgentEmployer(object):
    def __init__(self):

        # agent action structure
        self.policyType = None
        
        # state
        self.state = None
        
        # action
        self.action = None

        # Flags where agent is an RL model (True) or part of the environment (False)
        self.agent_flag = None

# properties of agent entities
class AgentPlanner(object):
    def __init__(self):

        # agent action structure
        self.policyType = None
        
        # state 
        self.state              = None
        
        # action
        self.action             = None

        # Flags where agent is an RL model (True) or part of the environment (False)
        self.agent_flag         = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)

        # Object classes in the simulation
        self.agents                     = None # Each entry is an agent in the simulation 
        self.allpopulationinformation   = None # ALL Customer related loan package information (for every module)
        self.trackingindicators         = None

        # Parameters of the simulation
        self.k_cycle            = None                                                  
        self.save_dta_flag      = None
        self.N_geog             = None	                                    
        self.feature_increment  = None                                            
        self.agent_config       = None                                         
        self.store_path         = None                                           
        self.tuition            = None
        self.episode_length     = None                                              
        self.sample_sz          = None
        self.init_params        = None
        
    # Sigmoid function
    def sig(self, x):
        return 1 / (1 + np.exp(-x))
    
    # update state of the world
    def step(self, iter_num):

        # ------------------Apply Decisions-----------------
        # Apply employer salary decisions
        self.apply_emp_actions()

        # Apply university budget allocation decisions
        self.apply_univ_budget_actions()

        # Episode failed if university can't pay faculty (if it does not fail, we continue environment updates)
        EnvTrackers = self.trackingindicators.EnvTrackers
        failed_ep = EnvTrackers["UNIV_fail_flag"]
        if not failed_ep:

            # Apply university admission decisions
            self.apply_univ_adm_actions()

            # Apply planner budget allocation decisions
            self.apply_planner_actions()

            # Update environment (or do this in Env script?) (This should handle queueing stuff an swapping out agents to be subsumed into the environment)
            failed_ep = self.update_env()

        # ------------------Regular State Information-----------------
        # Update university admissions agent state
        self.update_univ_adm_agent_state(iter_num)
        state0 = self.agents[0].state

        # Update university budget agent state
        self.update_univ_budget_agent_state(iter_num)
        state1 = self.agents[1].state

        # Update emploter agent state
        self.update_emp_agent_state(iter_num)
        state2 = self.agents[2].state

        # Update planner agent state
        self.update_planner_agent_state(iter_num)
        state3 = self.agents[3].state

        # Observations to return
        obs = [state0, state1, state2, state3]

        # Rewards to return
        EnvTrackers = self.trackingindicators.EnvTrackers
        EnvTrackers = self.UpdateRewardsTrackers(EnvTrackers)
        self.trackingindicators.EnvTrackers = EnvTrackers
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

        # Update profitability of each employee based on diversity program
        X_global, df_global, EnvTrackers = self.ProfitabilityUpdate(X_global, df_global, EnvTrackers)

        # Tertiary Investment
        X_global, df_global, EnvTrackers = self.TertiaryInvestmentFeatures(X_global, df_global, EnvTrackers)

        # (Environment update) Update employer profits
        X_global, df_global, EnvTrackers = self.EmployerProfitsCalc(X_global, df_global, EnvTrackers)

        # Episode failed if employer can't remain profitable
        failed_ep = EnvTrackers["EMP_fail_flag"]
        if failed_ep:
            return failed_ep

        # (Environment update) Update university budget to include tuition
        X_global, df_global, EnvTrackers = self.UNIVBudgetTuitionUpdate(X_global, df_global, EnvTrackers)

        # (Environment update) Update GPA of students in college
        X_global, df_global, EnvTrackers = self.GPAUpdate(X_global, df_global, EnvTrackers)

        # (Environment update) Determine who leaves or stays in college (if you leave before k=4 cycles, drop out of undergrad. If you leave between 4<=k<6 get undergrad but not master's. If you leave between 6<=k<9 get master's but not Ph.D
        X_global, df_global, EnvTrackers = self.TimeSpentUpdate(X_global, df_global, EnvTrackers)

        # (Environment update) Update peron's  features, remove them from the simulation, and place them back into the global population for sampling the next generation (After T time steps for one person)
        X_global, df_global, EnvTrackers = self.UpdateNextGen(X_global, df_global, EnvTrackers)

        # Store back
        df_full["global"]                   = df_global
        X_full["global"]                    = X_global

        self.allpopulationinformation.X_full    = X_full
        self.allpopulationinformation.df_full   = df_full
        self.trackingindicators.EnvTrackers     = EnvTrackers

        return failed_ep

#----------
    def apply_univ_adm_actions(self):

        agents          = self.agents
        EnvTrackers     = self.trackingindicators.EnvTrackers
        action_univ_adm = agents[0].action
        df_full         = self.allpopulationinformation.df_full
        X_full          = self.allpopulationinformation.X_full
        AppliedStudents     = EnvTrackers["Demographic_applied_track"]
        AdmittedStudents    = EnvTrackers["Demographic_admitted_track"]

        if action_univ_adm is not None:

            # Grab global variables
            df_global         = df_full["global"]
            X_global          = X_full["global"]

            tuition             = EnvTrackers["tuition"]
            N_total_units       = EnvTrackers["current_total_units"]
            N_students          = EnvTrackers["current_college_student"]
            c_student           = EnvTrackers["c_student"]
            feature_start_idxs  = EnvTrackers["feature_start_idxs"]
            norm_params         = EnvTrackers["norm_params"]
            
            # Get students who applied for school
            X_grad_high, df_grad_high = self.Global2AgentFeatures(X_global, df_global, EnvTrackers, "univ_adm", version="full")        

            # Number applied in this round
            N_high_grad = X_grad_high.shape[0]

            # Admit them either to college or pipeline them to workforce. Number admitted based on infrastruct/staff available. They want to fill up classes to make money!
            N = len(action_univ_adm)
            N_available = np.max( [np.floor(c_student*N_total_units-N_students), 0] ).astype(int)
            N_admit = min([N_high_grad, N_available])
            idx_admit = np.vstack((action_univ_adm, np.linspace(0,N-1,N)))
            idx_admit = idx_admit[:,idx_admit[0,:].argsort()[::-1]]
            idx_admit[0,:N_admit]=1
            idx_admit[0,N_admit:]=0
            idx_admit = idx_admit[:,idx_admit[1,:].argsort()]
            idx_admit = idx_admit[0,:].astype(bool)
            idx_reject = (idx_admit==0).astype(bool)

            # Admitted enter university (provide them with tuition)
            df_grad_high["INUNIV"][idx_admit]                           = 1
            X_grad_high[idx_admit, feature_start_idxs["INUNIV"][1]]     = 1
            df_grad_high["ANNUALTUIT"][idx_admit]                       = tuition
            X_grad_high[idx_admit, feature_start_idxs["ANNUALTUIT"][1]] = df_grad_high["ANNUALTUIT"][idx_admit]/norm_params["const_div"][feature_start_idxs["ANNUALTUIT"][0]]
            df_grad_high["TIMEINUNIV"][idx_admit]                       = -1
            X_grad_high[idx_admit, feature_start_idxs["TIMEINUNIV"][1]] = df_grad_high["TIMEINUNIV"][idx_admit]

            # Rejected immediately go to workforce
            df_grad_high['INWORKF'][idx_reject]                         = 1
            X_grad_high[idx_reject, feature_start_idxs['INWORKF'][1]]   = 1
            df_grad_high["TIMEINWORKF"][idx_reject]                       = -1
            X_grad_high[idx_reject, feature_start_idxs["TIMEINWORKF"][1]] = df_grad_high["TIMEINWORKF"][idx_reject]/norm_params["const_div"][feature_start_idxs["TIMEINWORKF"][0]]

            # Calculate number of students admitted to college
            Admitted_arr = np.zeros((2,1))
            for n_dem in range(2):
                Admitted_arr[n_dem] = sum( np.logical_and(df_grad_high["MINRTY"]==n_dem, idx_admit) )
            AdmittedStudents = np.hstack((AdmittedStudents,Admitted_arr))

            # Calculate number of people who applied
            Applied_arr = np.zeros((2,1))
            for n_dem in range(2):
                Applied_arr[n_dem] = sum( df_grad_high["MINRTY"]==n_dem )
            AppliedStudents = np.hstack((AppliedStudents,Applied_arr))

            # Update full feature set
            X_global, df_global                 = self.AgentFeaturesInsert2Global(X_global, df_global, X_grad_high, df_grad_high, "univ_adm", version="full")

            # Store back
            df_full["global"]                   = df_global
            X_full["global"]                    = X_global

            self.allpopulationinformation.X_full    = X_full
            self.allpopulationinformation.df_full   = df_full
            self.agents                             = agents

        else:

            # Calculate number of students admitted to college
            Admitted_arr = np.zeros((2,1))
            AdmittedStudents = np.hstack((AdmittedStudents,Admitted_arr))

            # Calculate number of people who applied
            Applied_arr = np.zeros((2,1))
            AppliedStudents = np.hstack((AppliedStudents,Applied_arr))

        # Update environmental trackers
        EnvTrackers["Demographic_applied_track"]        = AppliedStudents
        EnvTrackers["Demographic_admitted_track"]       = AdmittedStudents
        self.trackingindicators.EnvTrackers     = EnvTrackers

            
    def update_univ_adm_agent_state(self, t):

        agents          = self.agents
        df_full         = self.allpopulationinformation.df_full
        X_full          = self.allpopulationinformation.X_full
        EnvTrackers     = self.trackingindicators.EnvTrackers

        # Update State Information University Admissions Agent
        X_full, df_full = self.State_Update(X_full, df_full, EnvTrackers, "univ_adm", t+1, self.k_cycle["univ_adm"])
        agents[0].state = X_full["univ_adm"]

        self.agents                             = agents
        self.allpopulationinformation.df_full   = df_full
        self.allpopulationinformation.X_full    = X_full


    def apply_univ_budget_actions(self):

        agents                  = self.agents
        EnvTrackers             = self.trackingindicators.EnvTrackers
        action_univ_budget      = agents[1].action
        df_full                 = self.allpopulationinformation.df_full
        X_full                  = self.allpopulationinformation.X_full
        
        # Grab global variables
        df_global         = df_full["global"]
        X_global          = X_full["global"]
        
        if action_univ_budget is not None:

            # Budget for the university given current resources
            univ_budget_allocated = EnvTrackers['each_investment']["univ_budget"]

            # Get environment trackers
            UNIV_budget         = EnvTrackers["UNIV_budget_rollover"] + univ_budget_allocated
            N_faculty           = EnvTrackers["number_total_faculty_track"][-1]
            cost_faculty        = EnvTrackers["cost_faculty"]
                
            # Remainder of budget allocated to other things
            diversity_programs      = UNIV_budget*action_univ_budget[0]
            minority_scholarships   = UNIV_budget*action_univ_budget[1]
            majority_scholarships   = UNIV_budget*action_univ_budget[2]
            infra_budget            = UNIV_budget*action_univ_budget[3]
            salaries                = UNIV_budget*action_univ_budget[4]
            rollover                = UNIV_budget*action_univ_budget[5]
    #        print("UNIV Budget: "+str(action_univ_budget))

#            diversity_programs      = 0
#            minority_scholarships   = 0
#            majority_scholarships   = 0
#            infra_budget            = 0
#            salaries                = 0
#            rollover                = 0
#            UNIV_fail_flag = False
#            EnvTrackers["UNIV_fail_flag"] = UNIV_fail_flag
            
            # If university can't pay faculty, then episode fails
#            print("University Net Budget: "+str(salaries-N_faculty*cost_faculty))
            if salaries-N_faculty*cost_faculty<0:
                UNIV_fail_flag = True
            else:
                UNIV_fail_flag = False
            EnvTrackers["UNIV_fail_flag"] = UNIV_fail_flag

        else:

            # Remainder of budget allocated to other things
            diversity_programs      = None
            minority_scholarships   = None
            majority_scholarships   = None
            infra_budget            = None
            rollover                = None
            UNIV_fail_flag          = False
            EnvTrackers["UNIV_fail_flag"] = UNIV_fail_flag

        if not UNIV_fail_flag:

            # Determine minority students that get mentorship program
            X_global, df_global, EnvTrackers = self.update_mentees(X_global, df_global, EnvTrackers, diversity_programs)

            # Apply minority scholarship effect
            X_global, df_global, EnvTrackers = self.apply_scholarship(X_global, df_global, EnvTrackers, minority_scholarships, 'minority')

            # Apply general student scholarship effect (or majority)
            X_global, df_global, EnvTrackers = self.apply_scholarship(X_global, df_global, EnvTrackers, majority_scholarships, 'majority')

            # Apply infrastructure (and faculty) investment
            X_global, df_global, EnvTrackers, rollover = self.UpdateUnivInfrastructure(X_global, df_global, EnvTrackers, infra_budget, rollover)

            if rollover is not None:
                EnvTrackers["UNIV_budget_rollover"] = rollover

        # Store back
        df_full["global"]                   = df_global
        X_full["global"]                    = X_global

        self.allpopulationinformation.X_full    = X_full
        self.allpopulationinformation.df_full   = df_full
        self.agents                             = agents
        self.trackingindicators.EnvTrackers     = EnvTrackers


    def update_univ_budget_agent_state(self, t):

        agents          = self.agents
        df_full         = self.allpopulationinformation.df_full
        X_full          = self.allpopulationinformation.X_full
        EnvTrackers     = self.trackingindicators.EnvTrackers

        # Update State Information University Budget Agent
        X_full, df_full = self.State_Update(X_full, df_full, EnvTrackers, "univ_budget", t+1, self.k_cycle["univ_budget"])
        agents[1].state = X_full["univ_budget"]

        self.agents                             = agents
        self.allpopulationinformation.df_full   = df_full
        self.allpopulationinformation.X_full    = X_full


    def apply_emp_actions(self):

        agents                          = self.agents
        action_emp                      = agents[2].action
        EnvTrackers                     = self.trackingindicators.EnvTrackers
        df_full                         = self.allpopulationinformation.df_full
        X_full                          = self.allpopulationinformation.X_full
        Demographic_salary_track        = EnvTrackers['Demographic_salary_track']
        Demographic_in_workforce_track  = EnvTrackers['Demographic_in_workforce_track']

        if action_emp is not None:

            # Grab global variables
            df_global           = df_full["global"]
            X_global            = X_full["global"]
            feature_start_idxs  = EnvTrackers["feature_start_idxs"]
            norm_params         = EnvTrackers["norm_params"]

            # Get people employeed
            X_emp, df_emp = self.Global2AgentFeatures(X_global, df_global, EnvTrackers, "emp", version="full")

            # Set Salaries
            df_emp['SALARY']                            = action_emp
#            df_emp['SALARY']                            = np.ones(action_emp.shape)*30000
            df_emp['AVE_SALARY']                        = (df_emp['AVE_SALARY']*df_emp['TIMEINWORKF']+action_emp)/(df_emp['TIMEINWORKF']+1)
            X_emp[:,feature_start_idxs['SALARY'][1]]    = (df_emp['SALARY'] \
                                                            -norm_params["mu_test"][feature_start_idxs["SALARY"][0]])/norm_params["sig_test"][feature_start_idxs["SALARY"][0]]
            X_emp[:,feature_start_idxs['AVE_SALARY'][1]]= (df_emp['AVE_SALARY'] \
                                                            -norm_params["mu_test"][feature_start_idxs["SALARY"][0]])/norm_params["sig_test"][feature_start_idxs["SALARY"][0]]
            
            # Update tracked variables

            # Accumulated Salary
            Total_salar_arr = np.zeros((2,1))
            for n_dem in range(2):
                idx_dem_worforce        = df_emp["MINRTY"]==n_dem
                Total_salar_arr[n_dem]  = sum( df_emp[idx_dem_worforce]['SALARY'] )
            Demographic_salary_track    = np.hstack((Demographic_salary_track, Total_salar_arr))

            # Total people in workforce
            Total_workforce_arr = np.zeros((2,1))
            for n_dem in range(2):
                idx_dem_worforce            = df_emp["MINRTY"]==n_dem
                Total_workforce_arr[n_dem]  = sum( idx_dem_worforce )
            Demographic_in_workforce_track  = np.hstack((Demographic_in_workforce_track, Total_workforce_arr))

            # Update full feature set
            X_global, df_global = self.AgentFeaturesInsert2Global(X_global, df_global, X_emp, df_emp, "emp", version="full")


            # Store back
            df_full["global"]                   = df_global
            X_full["global"]                    = X_global
        
            self.allpopulationinformation.df_full   = df_full
            self.allpopulationinformation.X_full    = X_full

        else:

            # Accumulated Salary
            Total_salar_arr = np.zeros((2,1))
            Demographic_salary_track    = np.hstack((Demographic_salary_track, Total_salar_arr))

            # Total people in workforce
            Total_workforce_arr = np.zeros((2,1))
            Demographic_in_workforce_track  = np.hstack((Demographic_in_workforce_track, Total_workforce_arr))


        # Update tracked environmental parameters
        EnvTrackers['Demographic_salary_track']         = Demographic_salary_track
        EnvTrackers['Demographic_in_workforce_track']   = Demographic_in_workforce_track
        self.trackingindicators.EnvTrackers             = EnvTrackers

        #------------

    def update_emp_agent_state(self, t):

        agents          = self.agents
        df_full         = self.allpopulationinformation.df_full
        X_full          = self.allpopulationinformation.X_full
        EnvTrackers     = self.trackingindicators.EnvTrackers

        # Update State Information Employer Agent
        X_full, df_full = self.State_Update(X_full, df_full, EnvTrackers, "emp", t+1, self.k_cycle["emp"])
        agents[2].state = X_full["emp"]

        self.agents                             = agents
        self.allpopulationinformation.df_full   = df_full
        self.allpopulationinformation.X_full    = X_full  



    def apply_planner_actions(self):

        agents              = self.agents
        N_geog              = self.N_geog
        EnvTrackers         = self.trackingindicators.EnvTrackers
        action_planner      = agents[3].action
        norm_params         = EnvTrackers["norm_params"]
        feature_start_idxs  = EnvTrackers["feature_start_idxs"]
        df_full             = self.allpopulationinformation.df_full
        X_full              = self.allpopulationinformation.X_full

        # Grab global variables
        df_global         = df_full["global"]
        X_global          = X_full["global"]

        if action_planner is not None:

            # Get planner's budget and allocate it each time planner has an action to take
            Planner_budget    = EnvTrackers["Planner_budget"]

    #        print("PLANNER Budget: "+str(action_planner))
            
            # Get investment actions for planner
            investments_provided = action_planner # percentages that sum to 1

            # Break down and restructure subsidies        
            which_investment = ["tertiary", "univ_budget", "emp"]
            each_investment    = {"tertiary":np.array([]),
                               "univ_budget":investments_provided[0]*Planner_budget,
                                "emp":investments_provided[1]*Planner_budget}
    
            # Percentage of budget disbursed throughout current time step
            investment_percent = investments_provided[2]
            curr_investment_name = "tertiary"
            for n_g in range(N_geog):
                investment_percent_geog = investment_percent*investments_provided[int(n_g+3)]
                curr_investment = investment_percent_geog*Planner_budget
                each_investment [curr_investment_name] = np.append(each_investment [curr_investment_name], curr_investment)

#            which_investment = ["tertiary", "univ_budget", "emp"]
#            each_investment    = {"tertiary":np.ones(N_geog)*0,
#                               "univ_budget":Planner_budget,
#                                "emp":0}

        else:
            each_investment    = {"tertiary":None, "univ_budget":None, "emp":None}

        # Update environment trackers
        EnvTrackers['each_investment'] = each_investment
        
        # Store back
        df_full["global"]                       = df_global
        X_full["global"]                        = X_global

        self.allpopulationinformation.X_full    = X_full
        self.allpopulationinformation.df_full   = df_full
        self.agents                             = agents
        self.trackingindicators.EnvTrackers     = EnvTrackers


    def update_planner_agent_state(self, t):

        agents          = self.agents
        df_full         = self.allpopulationinformation.df_full
        X_full          = self.allpopulationinformation.X_full
        EnvTrackers     = self.trackingindicators.EnvTrackers

        # Update State Information Planner Agent
        X_full, df_full = self.State_Update(X_full, df_full, EnvTrackers, "planner", t+1, self.k_cycle["planner"])
        agents[3].state = X_full["planner"]

        self.agents                             = agents
        self.allpopulationinformation.df_full   = df_full
        self.allpopulationinformation.X_full    = X_full

#-------Functions to add:-------

    def AgentFeaturesInsert2Global(self, X_global, df_global, X_agent, df_agent, which_agent, version="full", idx=None):

        # High school graduates state/feature information
        if which_agent=="tertiary":
            if idx is not None:
                idx_tertiary                                            = idx
            else:
                idx_tertiary                                            = ~(df_global['INENV']==1)

            if version == "full":
                df_global[idx_tertiary]                                 = df_agent
                X_global[idx_tertiary]                                  = X_agent
            else:
                remove_feat_idxs                                        = np.array([var=="UGGPA" or var=="AVE_SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="N_UNIV_UNITS" or var=="N_FACULTY" or var=="N_STUDENTS_CURR" or var=="AGE" or var=="ANNUALTUIT" or var=="PLANBUDGET" or var=="UNIVBUDGET" or var=="CURRENTGPA" for var in feature_names	    ])
                remove_feat_idxs_preproc                                = np.array([var=="UGGPA" or var=="AVE_SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="N_UNIV_UNITS" or var=="N_FACULTY" or var=="N_STUDENTS_CURR" or var=="AGE" or var=="ANNUALTUIT" or var=="PLANBUDGET" or var=="UNIVBUDGET" or var=="CURRENTGPA" for var in feature_names_preproc ])
                df_global.loc[idx_tertiary, ~remove_feat_idxs]          = df_agent
                X_global.loc[idx_tertiary, ~remove_feat_idxs_preproc]   = X_agent

            return X_global, df_global


        # High school graduates state/feature information
        if which_agent == "univ_adm":
            if idx is not None:
                idx_univ_adm                                            = idx
            else:
                idx_univ_adm                                            = np.logical_and(df_global['TIMEINENV'] == 0, df_global['INENV']==1)
            if version == "full":
                df_global[idx_univ_adm]                                 = df_agent
                X_global[idx_univ_adm]                                  = X_agent
            else:
                remove_feat_idxs                                        = np.array([var=="AVE_SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="CURRENTGPA" or var=="AGE" or var=="PLANBUDGET" for var in feature_names         ])
                remove_feat_idxs_preproc                                = np.array([var=="AVE_SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="CURRENTGPA" or var=="AGE" or var=="PLANBUDGET" for var in feature_names_preproc ])
                df_global.loc[idx_univ_adm, ~remove_feat_idxs]          = df_agent
                X_global.loc[idx_univ_adm, ~remove_feat_idxs_preproc]   = X_agent

            return X_global, df_global
        
        # In university
        if which_agent == "univ_budget":
            if idx is not None:
                idx_univ_budget                                         = idx
            else:
                idx_univ_budget                                         = np.logical_and(df_global['TIMEINUNIV'] == 0, df_global['INUNIV']==1)

            if version == "full":
                df_global[idx_univ_budget]                              = df_agent
                X_global[idx_univ_budget]                               = X_agent
            else:
                remove_feat_idxs                                        = np.array([var=="AVE_SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="CURRENTGPA" or var=="AGE" or var=="PLANBUDGET" for var in feature_names         ])
                remove_feat_idxs_preproc                                = np.array([var=="AVE_SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="CURRENTGPA" or var=="AGE" or var=="PLANBUDGET" for var in feature_names_preproc ])
                df_global.loc[idx_univ_budget, ~remove_feat_idxs]       = df_agent
                X_global.loc[idx_univ_budget, ~remove_feat_idxs_preproc]= X_agent


            return X_global, df_global

        # Employer state/feature information
        if which_agent == "emp":
            if idx is not None:
                idx_in_workforce                                        = idx
            else:
                idx_in_workforce                                        = df_global['INWORKF'] == 1

            if version == "full":
                df_global[idx_in_workforce]                             = df_agent
                X_global[idx_in_workforce]                              = X_agent
            else:
                remove_feat_idxs                                        = np.array([var=="AVE_SALARY" or var=="SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="N_UNIV_UNITS" or var=="N_FACULTY" or var=="N_STUDENTS_CURR" or var=="ANNUALTUIT" or var=="PLANBUDGET" or var=="UNIVBUDGET" or var=="CURRENTGPA" for var in feature_names         ])
                remove_feat_idxs_preproc                                = np.array([var=="AVE_SALARY" or var=="SALARY" or var=="INWORKF" or var=="INUNIV" or var=="INENV" or var=="N_UNIV_UNITS" or var=="N_FACULTY" or var=="N_STUDENTS_CURR" or var=="ANNUALTUIT" or var=="PLANBUDGET" or var=="UNIVBUDGET" or var=="CURRENTGPA" for var in feature_names_preproc ])
                df_global.loc[idx_in_workforce, ~remove_feat_idxs]      = df_agent
                X_global.loc[idx_in_workforce, ~remove_feat_idxs_preproc]= X_agent

            return X_global, df_global


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


#----------

    def update_mentees(self, X, df, EnvTrackers, program_budget):

        # Scholarships that help minorities
        if program_budget is not None:

            feature_start_idxs      =   EnvTrackers["feature_start_idxs"]

            # Get environment tracking information
            cost_program        = EnvTrackers["cost_program"] # cost to finance one person through program

            # Get all students in university
            X_budget, df_budget = self.Global2AgentFeatures(X, df, EnvTrackers, "univ_budget", version="full")

            # Of all students get those who are just admitted minority students
            candidate_idxs      = df_budget['MINRTY'] == 1
            X_adm_minority      = X_budget[candidate_idxs]
            df_adm_minority     = df_budget[candidate_idxs]
            N_candidates        = sum(candidate_idxs)

            # Only if there are minorities just admitted to school
            if N_candidates > 0:

                # Probabilities for sampling who gets selected to enter program
                probabilities = (df_adm_minority['UGLOAN']/(df_adm_minority['SALARY']+1e-8)).to_numpy().astype(np.float)
                probabilities = ((probabilities+1e-10)/(sum(probabilities)+1e-8))
                probabilities = probabilities/sum(probabilities)

                # Sample who is in the program
                samp_idxs   = np.linspace(0,N_candidates-1,N_candidates)
                N_program   = int(np.min([N_candidates, np.floor(program_budget/cost_program)]))
                adm_idxs    = np.random.choice(samp_idxs, N_program, replace=False, p=probabilities).astype(int)

                # Update features of those admitted to program
                df_adm_minority['INMINTYPGRM'].iloc[adm_idxs] = 1
                X_adm_minority[adm_idxs, feature_start_idxs['INMINTYPGRM'][1]] = 1

                # Put this information back into the general dataframe
                X_budget[candidate_idxs]       = X_adm_minority
                df_budget[candidate_idxs]      = df_adm_minority
                X, df = self.AgentFeaturesInsert2Global(X, df, X_budget, df_budget, "univ_budget", version="full")

        return X, df, EnvTrackers


    def apply_scholarship(self, X, df, EnvTrackers, amount, scholarship_type):

        if amount is not None:

            # Get environment tracking information
            feature_start_idxs  = EnvTrackers["feature_start_idxs"]
            norm_params         = EnvTrackers["norm_params"]
            k_cycle             = EnvTrackers["k_cycle"]["univ_budget"]

            # Get students that may potentially receive scholarships
            X_budget, df_budget   = self.Global2AgentFeatures(X, df, EnvTrackers, "univ_budget", version="full")

            # Scholarships that help minorities
            if scholarship_type == 'minority':

                candidate_idxs      = df_budget['MINRTY'] == 1
                X_minority          = X_budget[candidate_idxs]
                df_minority         = df_budget[candidate_idxs]
                N_minority          = sum(candidate_idxs)
                
                if N_minority > 0:

                    tuition_minority    = df_minority['ANNUALTUIT']
                    amount              = amount/8          # (Undergrad and Master's tuition. The factor of 8 accounts for the fact that tuition doubles in Master's)

                    # Weighting feature
                    weights = ((df_minority['UGLOAN']+1e-10)/(df_minority['SALARY']+1e-8)).to_numpy().astype(np.float)
                    weights = (weights/sum(weights))
                    tuition_calc = self.adjust_tuition(df_minority, tuition_minority, weights, amount)

                    # Update Tuitions
                    df_minority['ANNUALTUIT'] = tuition_calc
                    X_minority[:, feature_start_idxs['ANNUALTUIT'][1]] = df_minority['ANNUALTUIT']/norm_params['const_div'][feature_start_idxs['ANNUALTUIT'][0]]

                    # Update University Student Info
                    X_budget[candidate_idxs]  = X_minority
                    df_budget[candidate_idxs] = df_minority

            # General scholarships (non-minority students)
            if scholarship_type == 'majority':

                candidate_idxs      = df_budget['MINRTY'] == 0
                X_majority          = X_budget[candidate_idxs]
                df_majority         = df_budget[candidate_idxs]
                N_majority          = sum(candidate_idxs)
                
                if N_majority > 0:

                    tuition_majority    = df_majority['ANNUALTUIT']
                    amount              = amount/8          # (Undergrad and Master's tuition. The factor of 8 accounts for the fact that tuition doubles in Master's)

                    # Weighting feature
                    weights = ((df_majority['UGLOAN']+1e-10)/(df_majority['SALARY']+1e-8)).to_numpy().astype(np.float)
                    weights = (weights/sum(weights))
                    tuition_calc = self.adjust_tuition(df_majority, tuition_majority, weights, amount)

                    # Update Tuitions
                    df_majority['ANNUALTUIT'] = tuition_calc
                    X_majority[:, feature_start_idxs['ANNUALTUIT'][1]] = df_majority['ANNUALTUIT']/norm_params['const_div'][feature_start_idxs['ANNUALTUIT'][0]]

                    # Update University Student Info
                    X_budget[candidate_idxs]  = X_majority
                    df_budget[candidate_idxs] = df_majority

            # Put this information back into the general dataframe
            X, df = self.AgentFeaturesInsert2Global(X, df, X_budget, df_budget, "univ_budget", version="full")
            
        return X, df, EnvTrackers

    
    def adjust_tuition(self, df, tuitions_offered, weights, scholarship):

        N_samp,_ = df.shape

        # Order from poorest to wealthiest when applying subsidies
        idxs_ordered = np.linspace(0,N_samp-1,N_samp)
        adjusted_tuition_arr = np.vstack((idxs_ordered,tuitions_offered,weights))
        adjusted_tuition_arr = adjusted_tuition_arr[:,np.argsort(adjusted_tuition_arr[-1,:])[::-1]]
        
        left_over=0

        # Update preimum for each person (one at a time because of poor-over effect
        for nn in range(N_samp):

            # Adjust premium for investment (use up all left over investment on next poorest person)
            adjusted_tuition = adjusted_tuition_arr[1,nn] - adjusted_tuition_arr[2,nn]*scholarship - left_over
            left_over = 0

            # If investment greater than premium, treat as left over money to poor over
            # for the next person's investment
            if adjusted_tuition<0:
                left_over += -adjusted_tuition
                adjusted_tuition = 0

            # Apply premium
            adjusted_tuition_arr[1,nn] = adjusted_tuition

        # Reorder back to orignal sort
        adjusted_tuition_arr = adjusted_tuition_arr[:,np.argsort(adjusted_tuition_arr[0,:])]
            
        return adjusted_tuition_arr[1,:]


    def UpdateUnivInfrastructure(self, X, df, EnvTrackers, infra_budget, rollover):

        base_cost, prop_cost            = EnvTrackers["university_infra_cost_params"]
        constructing_units              = EnvTrackers["constructing_units"]
        current_total_units             = EnvTrackers["current_total_units"]
        number_total_units_track        = EnvTrackers["number_total_units_track"]
        constructing_time               = EnvTrackers["constructing_time"]
        prop_time, base_time            = EnvTrackers["time_coeffs_infra_construct"]
        deterioration_univ_infra        = EnvTrackers["deterioration_univ_infra"]
        feature_start_idxs              = EnvTrackers["feature_start_idxs"]
        norm_params                     = EnvTrackers['norm_params']
        c_faculty                       = EnvTrackers["c_faculty"]
        number_total_faculty_track      = EnvTrackers["number_total_faculty_track"]

        # Add new projects for infrastructure added to university
        if infra_budget is not None:

            N_new_units = int(np.max([np.floor((infra_budget-base_cost)/prop_cost),0]))                   # number of beds invested in this round
        
            if N_new_units> 0:
                rollover += infra_budget - (N_new_units*prop_cost + base_cost)                           # What goes unspent on beds gets rolled over
                constructing_units = np.append(constructing_units, N_new_units)
                time2build = np.round(prop_time*N_new_units + base_time)
                constructing_time = np.append(constructing_time,time2build)
            else:
                rollover += infra_budget

        else:
            rollover = None

        # Update time on all existing infrastructure plans being built
        constructing_time           -= 1
        finished_projects           = constructing_time==0
        N_new_units                 = sum(constructing_units[finished_projects])  # - deterioration_univ_infra <---- figure out how to incorporate this later
        constructing_units          = np.delete(constructing_units, finished_projects)  # Remove finished projects from constructing
        constructing_time           = np.delete(constructing_time, finished_projects)
        current_total_units         += N_new_units

        # Update tracking indicator variables
        
        # Update number of units at university based on finished projects
        df["N_UNIV_UNITS"]                              = current_total_units
        X[:, feature_start_idxs["N_UNIV_UNITS"][1]]     = current_total_units/norm_params["const_div"][feature_start_idxs["N_UNIV_UNITS"][0]]
        number_total_units_track                        = np.hstack((number_total_units_track,current_total_units))

        # Update number of faculty at university based on new infrastructure (c_faculty new faculty members per unit of infrastructure)
        current_total_faculty                       = c_faculty*current_total_units
        df["N_FACULTY"]                             = current_total_faculty
        X[:, feature_start_idxs["N_FACULTY"][1]]    = current_total_faculty/norm_params["const_div"][feature_start_idxs["N_FACULTY"][0]]
        number_total_faculty_track                  = np.hstack((number_total_faculty_track, current_total_faculty))

        # Update EnvTrackers and send back
        EnvTrackers["constructing_time"]            = constructing_time
        EnvTrackers["constructing_units"]           = constructing_units
        EnvTrackers["current_total_units"]          = current_total_units
        EnvTrackers["number_total_units_track"]     = number_total_units_track
        EnvTrackers["number_total_faculty_track"]   = number_total_faculty_track

        return X, df, EnvTrackers, rollover


#------

    def ProfitabilityUpdate(self, X, df, EnvTrackers):

        # Environmental variables
        diversity_investment = EnvTrackers['each_investment']["emp"] 
        feature_start_idxs   = EnvTrackers['feature_start_idxs']
        norm_params          = EnvTrackers['norm_params']

        # Adjust profitability of workforce to account for planner's diversity investment
        idx_min_workf   = np.logical_and(df["MINRTY"] == 1, df["INWORKF"] == 1)
        df_min_workf    = df[idx_min_workf]
        X_min_workf     = X[idx_min_workf]
        N_minority      = sum(idx_min_workf)

        # Apply diversity investment
        df_min_workf["DIVINVEST"] = diversity_investment/(N_minority+1e-10)
        X_min_workf[:, feature_start_idxs['DIVINVEST'][1]] = df_min_workf['DIVINVEST']/norm_params["const_div"][feature_start_idxs["DIVINVEST"][0]]

        # Update global variables
        df[idx_min_workf]   = df_min_workf
        X[idx_min_workf]    = X_min_workf

        return X, df, EnvTrackers

    

    def TertiaryInvestmentFeatures(self, X, df, EnvTrackers):

        N_geog                      = EnvTrackers["N_geog"]
        updateable_feature          = EnvTrackers["updateable_feature"]
        feature_polarity            = EnvTrackers["feature_polarity"]
        feature_extremes            = EnvTrackers["feature_extremes"]
        feature_increment           = EnvTrackers["feature_increment"]
        tertiary_investment         = EnvTrackers['each_investment']["tertiary"]
        prop_max_investment         = EnvTrackers["Planner_budget"]/(2*N_geog)
        deter_const                 = EnvTrackers["deterioration_public"]

        if tertiary_investment is not None:

            # Get tertiary population
            X_tertiary, df_tertiary = self.Global2AgentFeatures(X, df, EnvTrackers, "tertiary", version="full")

            # Update features of people
            for n_g in range(N_geog):

                df_curr_geog = df_tertiary[df_tertiary["REGION"]==n_g]
                X_curr_geog = X_tertiary[df_tertiary["REGION"]==n_g]
                curr_investment = tertiary_investment[n_g]


                # Select who gets positive and negative feature updates
                N_samp,_ = X_curr_geog.shape
                prob_calc = 0.3928055160151634+ 0.4*self.sig((curr_investment-deter_const-prop_max_investment*0.5)*4/(prop_max_investment*0.5))
                threshs = [np.clip(prob_calc,0,1), np.clip(prob_calc+0.2,0,1)]
                samp = uniform(np.zeros(df_curr_geog.shape[0]),np.ones(df_curr_geog.shape[0]))
                upgrade_samp_idx   = samp<=threshs[0]
                downgrade_samp_idx = samp>threshs[1]

                for n_f in range(len(updateable_feature)):

                    curr_feature            = updateable_feature[n_f]

                    # We update UGGPA later as a function of these feature updates
                    if curr_feature != "UGGPA":
                        curr_feature_extremes       = feature_extremes[curr_feature]
                        curr_feature_polarity       = feature_polarity[curr_feature]
                        df_curr_geog, X_curr_geog   = self.FeaturesUpdate(df_curr_geog, X_curr_geog, EnvTrackers, upgrade_samp_idx,
                                                               downgrade_samp_idx, curr_feature, curr_feature_extremes,
                                                               curr_feature_polarity, feature_increment)

                df_tertiary[df_tertiary["REGION"]==n_g] = df_curr_geog
                X_tertiary[df_tertiary["REGION"]==n_g] = X_curr_geog

            # Update population features
            X, df = self.AgentFeaturesInsert2Global(X, df, X_tertiary, df_tertiary, "tertiary", version="full")

        return X, df, EnvTrackers


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
            pdb.set_trace()
            # Update improved samples
            df[feature].iloc[upgrade_samp_idx] = min([df[feature].iloc[upgrade_samp_idx] + feature_polarity, feature_maximum])

            # Update deteriorated samples
            df[feature].iloc[downgrade_samp_idx]  = max([df[feature].iloc[downgrade_samp_idx] - feature_polarity, feature_minimum])

            pdb.set_trace()

            # How code below should work: Assume that all discrete features in the df are represented
            # by one column containing values from 0 to N_col-1 and
            # all discrete features in X are represented by N_col-1 columns of all zero except in the index
            # associated with the value represented in the df column (the one except is that value N_col-1 is represented by 0s in all columns)
            # Then, we update everything given these assumptions.
            pdb.set_trace()
            new_binaries = df[feature].to_numpy()
            X_update = X[:, start_idx:stop_idx+1]
            N_cols = X_update.shape
            vals = np.zeros((1,N_cols))
            pdb.set_trace()
            X_update[new_binaries==N_cols,:] = vals
            for ii in range(N_cols):
                vals_curr = vals.copy()
                vals_curr[ii] = 1
                X_update[new_binaries==ii,:] = vals_curr
            pdb.set_trace()
            X[:, start_idx:stop_idx+1] = X_update
            pdb.set_trace()

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



    def EmployerProfitsCalc(self, X, df, EnvTrackers):

        emp_profits_track               = EnvTrackers["emp_profits_track"]
        profitability_weights           = EnvTrackers["profitability_weights"]
        max_profitability               = EnvTrackers["max_profitability"]
        max_salary                      = EnvTrackers["feature_extremes"]['SALARY'][1]
        c_emp_div_incent                = EnvTrackers["c_emp_div_incent"]

        idx_workf       = np.logical_and(df["TIMEINWORKF"] >= 0, df["INWORKF"] == 1)
        X_workf         = X[idx_workf]
        df_workf        = df[idx_workf]
        
        # Get Employee Profitability
        gpa         = df_workf["UGGPA"]
        uni_exp     = df_workf["DGRDG"]
        age         = df_workf["AGE"]
        divinvest   = df_workf["DIVINVEST"]*c_emp_div_incent
        salaries    = df_workf["SALARY"]

        # Calculate value of each term
        GPA         = gpa/4
        DEGREE      = uni_exp/3
        AGE         = age/40
        SALARY      = np.clip((divinvest+salaries)/max_salary,0.15,1)
        COST        = np.clip((salaries)/max_salary,0.15,1)

        coeff = 3-3.3*( (1/3)*GPA + (1/3)*DEGREE  +  (1/3)*(AGE-AGE**2)   )
        emp_profits = sum(1.2*SALARY-coeff*(COST**2)+0.1)
#        emp_profits = sum((1.2*SALARY-coeff*(COST**2)+0.1)[df_workf['MINRTY']==1])

        # Update environment trackers
        
        # Total profits
        emp_profits_track = np.hstack((emp_profits_track, emp_profits))

        # If profits go negative, episode failed
        EMP_fail_flag = sum(emp_profits_track)<0

        # Update Environment Trackers
        EnvTrackers["EMP_fail_flag"]                    = EMP_fail_flag

        # If employer can't remain profitable, then episode fails
    #    print("Employer profits: "+str(sum(emp_profits_track)))

    #    if emp_profits>0:
    #        print("salaries: "+str(salaries.shape))
    #        plt.hist(salaries,bins=50)
    #        plt.show()
    #        pdb.set_trace()
            
        if EMP_fail_flag:
            return X, df, EnvTrackers
            
        EnvTrackers["emp_profits_track"]                = emp_profits_track

        return X, df, EnvTrackers


    def UNIVBudgetTuitionUpdate(self, X, df, EnvTrackers):

        rollover            = EnvTrackers["UNIV_budget_rollover"]
        norm_params         = EnvTrackers["norm_params"]
        feature_start_idxs  = EnvTrackers["feature_start_idxs"]

        # Budget for the university (currently only contains money from planner)
        univ_budget_allocated = EnvTrackers['each_investment']["univ_budget"]

        # Get students in college
        idx_in_univ = np.logical_and(df['TIMEINUNIV']>=0, df['INUNIV']==1)

        # Update budget
        univ_budget_allocated+=sum(df[idx_in_univ]['ANNUALTUIT'])

        # Update the university budget feature for making decisions next time step (include amount allocated by planner and rolled over by university.
        # No need to include accumulated tuition. This information is directly available from the ANNUALTUIT variable.)
        if rollover is not None:
            univ_budget_allocated += rollover

        # Incorporate update in EnvTrackers
        EnvTrackers['each_investment']["univ_budget"] = univ_budget_allocated
        df["UNIVBUDGET"] = EnvTrackers["each_investment"]["univ_budget"]
        X[:, feature_start_idxs['UNIVBUDGET'][1]] = df['UNIVBUDGET']/norm_params['const_div'][feature_start_idxs['UNIVBUDGET'][0]]

        return X, df, EnvTrackers


    def GPAUpdate(self, X, df, EnvTrackers):

        '''Update the GPA feature for everyone in the system at current time step.'''
        
        # Get environment parameter information
        Demographic_total_sum_gpa_track = EnvTrackers["Demographic_total_sum_gpa_track"]

        # Update GPAs of those in the system already
        idx_old_univ                            = np.logical_and( df['TIMEINUNIV']>0, df['INUNIV']==1 )
        X_old_univ                              = X[idx_old_univ]
        df_old_univ                             = df[idx_old_univ]
        X_old_univ, df_old_univ, EnvTrackers    = self.apply_GPA_update(X_old_univ, df_old_univ, EnvTrackers, update_flag='Old')

        # Get people just admitted and set their base GPA
        idx_new_univ                            = np.logical_and( df['TIMEINUNIV']==0, df['INUNIV']==1)
        X_new_univ                              = X[idx_new_univ]
        df_new_univ                             = df[idx_new_univ]
        X_new_univ, df_new_univ, EnvTrackers    = self.apply_GPA_update(X_new_univ, df_new_univ, EnvTrackers, update_flag='New')
        
        # Update environment trackers

        # Combine all GPAs from new and only students
        GPAs_this_year = np.zeros(X.shape[0])
        GPAs_this_year[idx_new_univ] = df_new_univ['CURRENTGPA']
        GPAs_this_year[idx_old_univ] = df_old_univ['CURRENTGPA']

        # Sum of everyone's GPAs that year 
        Total_sum_GPA_arr = np.zeros((2,1))
        for n_dem in range(2):
            idx_sum_GPA = np.logical_and(df["MINRTY"]==n_dem, df['INUNIV']==1)
            Total_sum_GPA_arr[n_dem] = sum(GPAs_this_year[idx_sum_GPA])
        Demographic_total_sum_gpa_track = np.hstack((Demographic_total_sum_gpa_track, Total_sum_GPA_arr))

        # Update population data
        df[idx_old_univ]    = df_old_univ
        X[idx_old_univ]     = X_old_univ
        df[idx_new_univ]    = df_new_univ
        X[idx_new_univ]     = X_new_univ

        # Update Environmental trackers
        EnvTrackers["Demographic_total_sum_gpa_track"]  = Demographic_total_sum_gpa_track
        
        return X, df, EnvTrackers


    def apply_GPA_update(self, X, df, EnvTrackers, prob_deviate=0.25, update_flag='New'):

        # Get environment parameter information
        norm_params         = EnvTrackers['norm_params']
        feature_start_idxs  = EnvTrackers['feature_start_idxs']
        feature_extremes    = EnvTrackers["feature_extremes"]
        min_gpa             = feature_extremes['UGGPA'][0]
        max_gpa             = feature_extremes['UGGPA'][1]

        if update_flag == 'New':

            # Get new-student-specific environment parameter information
            reg             = EnvTrackers["uggpa_regressor"]
            full_tuition    = EnvTrackers['tuition']

            N_features      = X.shape[1]
            delete_features = N_features-np.linspace(1,17,17, dtype=int)
            delete_features = np.append(delete_features, feature_start_idxs['UGGPA'][1])
            X_wo_UGGPA      = np.delete(X, delete_features, axis=1)

            # Predict person's GPA given their features
            if X_wo_UGGPA.shape[0]>0:
                curr_year_gpa_new_univ = reg.predict(X_wo_UGGPA)
            else:
                curr_year_gpa_new_univ = df['UGGPA']

            # Get initial GPA (upon entry to university...initial condition)
            tuition                     = df['ANNUALTUIT']
            in_program                  = df['INMINTYPGRM']
            gpa_rng                     = np.array([-0.4, 0.4])
            delta_rng_update            = 0.1*(1-tuition/full_tuition) + 0.3*in_program
            aa                          = gpa_rng[0] + delta_rng_update
            bb                          = gpa_rng[1] + delta_rng_update
            delta_GPA                   = uniform(aa, bb)
            curr_year_gpa_new_univ      += delta_GPA
            curr_year_gpa_new_univ      = np.clip(curr_year_gpa_new_univ, min_gpa, max_gpa)

            # Update current GPA (and renormalized)
            df['CURRENTGPA']                        = curr_year_gpa_new_univ
            X[:,feature_start_idxs['CURRENTGPA'][1]]= df['CURRENTGPA']/norm_params["const_div"][feature_start_idxs["CURRENTGPA"][0]]

            # Update cumulative GPA (and renormalized)
            df['UGGPA']                             = curr_year_gpa_new_univ
            X[:,feature_start_idxs['UGGPA'][1]]     = (df['UGGPA'] \
                                                                -norm_params["mu_test"][feature_start_idxs["UGGPA"][0]])/norm_params["sig_test"][feature_start_idxs["UGGPA"][0]]

            # Update Loans (and renormalized)
            df['UGLOAN']                            = 0
            X[:,feature_start_idxs['UGLOAN'][1]]     = (df['UGLOAN'] \
                                                                -norm_params["mu_test"][feature_start_idxs["UGLOAN"][0]])/norm_params["sig_test"][feature_start_idxs["UGLOAN"][0]]
            df['GRLOAN']                            = 0
            X[:,feature_start_idxs['GRLOAN'][1]]     = (df['GRLOAN'] \
                                                                -norm_params["mu_test"][feature_start_idxs["GRLOAN"][0]])/norm_params["sig_test"][feature_start_idxs["GRLOAN"][0]]

            # Update degree type (everyone starts with no degree)
            df['DGRDG']                            = 0
            X[:,feature_start_idxs['DGRDG'][1]]     = (df['DGRDG'] \
                                                                -norm_params["mu_test"][feature_start_idxs["DGRDG"][0]])/norm_params["sig_test"][feature_start_idxs["DGRDG"][0]]  

        else:

            # Likelihood that GPA deviates from current cummulative GPA
            one = np.ones(df['UGGPA'].shape)
            deviate_GPA = uniform(-0.25*one, 0.25*one)

            # Update current GPA (and renormalize)
            df['CURRENTGPA'] = np.clip( df['CURRENTGPA']+deviate_GPA, min_gpa, max_gpa )
            X[:,feature_start_idxs['CURRENTGPA'][1]] = df['CURRENTGPA']/norm_params["const_div"][feature_start_idxs["CURRENTGPA"][0]]

            # Update cummulative GPA (and renormalize)
            df['UGGPA'] = (df['UGGPA']*df['TIMEINUNIV'] + df['CURRENTGPA'] )/(df['TIMEINUNIV']+1)
            X[:,feature_start_idxs['UGGPA'][1]]  = (df['UGGPA'] \
                                                    -norm_params["mu_test"][feature_start_idxs["UGGPA"][0]])/norm_params["sig_test"][feature_start_idxs["UGGPA"][0]]

        return X, df, EnvTrackers


    def TimeSpentUpdate(self, X, df, EnvTrackers):

        feature_start_idxs                          = EnvTrackers['feature_start_idxs']
        norm_params                                 = EnvTrackers['norm_params']
        Demographic_in_univ_track                   = EnvTrackers["Demographic_in_univ_track"]
        Demographic_went_undergrad_track            = EnvTrackers["Demographic_went_undergrad_track"]
        Demographic_went_masters_track              = EnvTrackers['Demographic_went_masters_track']
        Demographic_went_doctorate_track            = EnvTrackers['Demographic_went_doctorate_track']
        Demographic_graduated_undergrad_track       = EnvTrackers['Demographic_graduated_undergrad_track']
        Demographic_graduated_masters_track         = EnvTrackers['Demographic_graduated_masters_track']
        Demographic_graduated_doctorate_track       = EnvTrackers['Demographic_graduated_doctorate_track']
        N_students                                  = EnvTrackers["current_college_student"]

        # Get those in workforce
        idx_emp = df['INWORKF'] == 1
        X_emp   = X[idx_emp]
        df_emp  = df[idx_emp]
        
        # Get those in school
        idx_all_univ    = df['INUNIV']==1
        X_all_univ      = X[idx_all_univ]
        df_all_univ     = df[idx_all_univ]
        idx_in_univ     = df_all_univ['TIMEINUNIV']>=0
        X_in_univ       = X_all_univ[idx_in_univ]
        df_in_univ      = df_all_univ[idx_in_univ]

        # Update decision to stay in school or leave

        # Undergraduate Updates
        Went_Undergrad                                                              = df_in_univ[df_in_univ['TIMEINUNIV']==0]
        Graduated_Undergrad                                                         = df_in_univ[df_in_univ['TIMEINUNIV']==3]
        idx_undergrad                                                               = np.logical_and( df_in_univ['TIMEINUNIV']>=0, df_in_univ['TIMEINUNIV']<4)
        df_undergrad                                                                = df_in_univ[idx_undergrad]
        X_undergrad                                                                 = X_in_univ[idx_undergrad]
        df_undergrad['UGLOAN']                                                      += df_undergrad['ANNUALTUIT']
        X_undergrad[:,feature_start_idxs['UGLOAN'][1]]                              = (df_undergrad['UGLOAN']
                                                                                           -norm_params["mu_test"][feature_start_idxs["UGLOAN"][0]])/norm_params["sig_test"][feature_start_idxs["UGLOAN"][0]]
        prob                                                                        = self.get_probability(df_undergrad, 'undergrad')
        one                                                                         = np.ones(prob.shape[0], dtype=int)
        Not_in_univ_undergrad                                                       = binomial(one, prob)
        Stay_in_univ_undergrad                                                      = np.abs(Not_in_univ_undergrad-1)
        idx_not_in_univ_undergrad                                                   = Not_in_univ_undergrad == 1
        idx_stay_in_univ_undergrad                                                  = ~idx_not_in_univ_undergrad
        df_undergrad['INUNIV']                                                      = Stay_in_univ_undergrad
        X_undergrad[:, feature_start_idxs['INUNIV'][1]]                             = Stay_in_univ_undergrad
        df_undergrad['INWORKF']                                                     = Not_in_univ_undergrad
        X_undergrad[:, feature_start_idxs['INWORKF'][1]]                            = Not_in_univ_undergrad
        df_undergrad['CURRENTGPA'][idx_not_in_univ_undergrad]                       = 0
        X_undergrad[idx_not_in_univ_undergrad, feature_start_idxs['CURRENTGPA'][1]] = 0
        df_undergrad['INMINTYPGRM'][idx_not_in_univ_undergrad]                      = 0
        X_undergrad[idx_not_in_univ_undergrad, feature_start_idxs['INMINTYPGRM'][1]]= 0
        df_undergrad['ANNUALTUIT'][idx_not_in_univ_undergrad]                       = 0
        X_undergrad[idx_not_in_univ_undergrad, feature_start_idxs['ANNUALTUIT'][1]] = 0

        # Master's Updates
        Went_Masters                                                                = df_in_univ[df_in_univ['TIMEINUNIV']==4]
        Graduated_Masters                                                           = df_in_univ[df_in_univ['TIMEINUNIV']==5]
        idx_masters                                                                 = np.logical_and(df_in_univ['TIMEINUNIV']>=4, df_in_univ['TIMEINUNIV']<6)
        df_masters                                                                  = df_in_univ[idx_masters]
        X_masters                                                                   = X_in_univ[idx_masters]
        df_masters['GRLOAN']                                                        += df_undergrad['ANNUALTUIT']*2
        X_undergrad[:,feature_start_idxs['GRLOAN'][1]]                              = (df_undergrad['GRLOAN']
                                                                                           -norm_params["mu_test"][feature_start_idxs["GRLOAN"][0]])/norm_params["sig_test"][feature_start_idxs["GRLOAN"][0]]
        prob                                                                        = self.get_probability(df_masters, 'masters')
        one                                                                         = np.ones(prob.shape[0], dtype=int)
        Not_in_univ_masters                                                         = binomial(one, prob)
        Stay_in_univ_masters                                                        = np.abs(Not_in_univ_masters-1)
        idx_not_in_univ_masters                                                     = Not_in_univ_masters == 1
        idx_stay_in_univ_masters                                                    = ~idx_not_in_univ_masters
        df_masters['INUNIV']                                                        = Stay_in_univ_masters
        X_masters[:, feature_start_idxs['INUNIV'][1]]                               = Stay_in_univ_masters
        df_masters['INWORKF']                                                       = Not_in_univ_masters
        X_masters[:, feature_start_idxs['INWORKF'][1]]                              = Not_in_univ_masters
        df_masters['CURRENTGPA'][idx_not_in_univ_masters]                           = 0
        X_masters[idx_not_in_univ_masters, feature_start_idxs['CURRENTGPA'][1]]     = 0
        df_masters['INMINTYPGRM'][idx_not_in_univ_masters]                          = 0
        X_masters[idx_not_in_univ_masters, feature_start_idxs['INMINTYPGRM'][1]]    = 0
        df_masters['ANNUALTUIT'][idx_not_in_univ_masters]                           = 0
        X_masters[idx_not_in_univ_masters, feature_start_idxs['ANNUALTUIT'][1]]     = 0


        # Ph.D Updates
        Went_Doctorate                                                              = df_in_univ[df_in_univ['TIMEINUNIV']==6]
        Graduated_Doctorate                                                         = df_in_univ[df_in_univ['TIMEINUNIV']==10]
        idx_doctorate                                                               = np.logical_and(df_in_univ['TIMEINUNIV']>=6, df_in_univ['TIMEINUNIV']<11)
        df_doctorate                                                                = df_in_univ[idx_doctorate]
        X_doctorate                                                                 = X_in_univ[idx_doctorate]
        prob                                                                        = self.get_probability(df_doctorate, 'doctorate')
        one                                                                         = np.ones(prob.shape[0], dtype=int)
        Not_in_univ_doctorate_1                                                     = binomial(one, prob)
        Not_in_univ_doctorate_2                                                     = df_doctorate['TIMEINUNIV']==10
        Not_in_univ_doctorate                                                       = np.logical_or(Not_in_univ_doctorate_1, Not_in_univ_doctorate_2)
        Stay_in_univ_doctorate                                                      = np.abs(Not_in_univ_doctorate-1)
        idx_not_in_univ_doctorate                                                   = Not_in_univ_doctorate == 1
        idx_stay_in_univ_doctorate                                                  = ~idx_not_in_univ_doctorate
        df_doctorate['INUNIV']                                                      = Stay_in_univ_doctorate
        X_doctorate[:, feature_start_idxs['INUNIV'][1]]                             = Stay_in_univ_doctorate
        df_doctorate['INWORKF']                                                     = Not_in_univ_doctorate
        X_doctorate[:, feature_start_idxs['INWORKF'][1]]                            = Not_in_univ_doctorate    
        df_doctorate['CURRENTGPA'][idx_not_in_univ_doctorate]                       = 0
        X_doctorate[idx_not_in_univ_doctorate, feature_start_idxs['CURRENTGPA'][1]] = 0
        df_doctorate['INMINTYPGRM'][idx_not_in_univ_doctorate]                      = 0
        X_doctorate[idx_not_in_univ_doctorate, feature_start_idxs['INMINTYPGRM'][1]]= 0
        df_doctorate['ANNUALTUIT'][idx_not_in_univ_doctorate]                       = 0
        X_doctorate[idx_not_in_univ_doctorate, feature_start_idxs['ANNUALTUIT'][1]] = 0

        # Update those that got degrees
        idx_undergrad_graduate                                                      = df_undergrad['TIMEINUNIV'] ==3
        df_undergrad['DGRDG'][idx_undergrad_graduate]                               = 1
        X_undergrad[idx_undergrad_graduate,feature_start_idxs['DGRDG'][1]]          = (df_undergrad['DGRDG'][idx_undergrad_graduate]
                                                                                           -norm_params["mu_test"][feature_start_idxs["DGRDG"][0]])/norm_params["sig_test"][feature_start_idxs["DGRDG"][0]]

        idx_masters_graduate                                                        = df_masters['TIMEINUNIV']==5
        df_masters['DGRDG'] [idx_masters_graduate]                                  = 2
        X_masters[idx_masters_graduate,feature_start_idxs['DGRDG'][1]]              = (df_masters['DGRDG'][idx_masters_graduate]
                                                                                           -norm_params["mu_test"][feature_start_idxs["DGRDG"][0]])/norm_params["sig_test"][feature_start_idxs["DGRDG"][0]]

        idx_doctorate_graduate                                                      = df_doctorate['TIMEINUNIV']==10
        df_doctorate['DGRDG'][idx_doctorate_graduate]                               = 3
        X_doctorate[idx_doctorate_graduate,feature_start_idxs['DGRDG'][1]]          = (df_doctorate['DGRDG'][idx_doctorate_graduate]
                                                                                           -norm_params["mu_test"][feature_start_idxs["DGRDG"][0]])/norm_params["sig_test"][feature_start_idxs["DGRDG"][0]]

        # --------Update variables----------    
        # Update university variables
        df_in_univ[idx_undergrad]   = df_undergrad
        df_in_univ[idx_masters]     = df_masters
        df_in_univ[idx_doctorate]   = df_doctorate
        X_in_univ[idx_undergrad]    = X_undergrad
        X_in_univ[idx_masters]      = X_masters
        X_in_univ[idx_doctorate]    = X_doctorate
        X_all_univ[idx_in_univ]     = X_in_univ
        df_all_univ[idx_in_univ]    = df_in_univ


        #----------Update EnvTrackers---------
        # Total graduated undergrad
        Graduate_undergrad_arr = np.zeros((2,1))
        for n_dem in range(2):
            idx_grad_undergrad                  = Graduated_Undergrad["MINRTY"]==n_dem
            Graduate_undergrad_arr[n_dem]       = sum( idx_grad_undergrad )
        Demographic_graduated_undergrad_track   = np.hstack((Demographic_graduated_undergrad_track, Graduate_undergrad_arr))

        # Total graduated masters
        Graduate_masters_arr = np.zeros((2,1))
        for n_dem in range(2):
            idx_grad_masters                    = Graduated_Masters["MINRTY"]==n_dem
            Graduate_masters_arr[n_dem]         = sum( idx_grad_masters )
        Demographic_graduated_masters_track     = np.hstack((Demographic_graduated_masters_track, Graduate_masters_arr))

        # Total graduated doctorate
        Graduate_doctorate_arr = np.zeros((2,1))
        for n_dem in range(2):
            idx_grad_doctorate                  = Graduated_Doctorate["MINRTY"]==n_dem
            Graduate_doctorate_arr[n_dem]       = sum( idx_grad_doctorate )
        Demographic_graduated_doctorate_track   = np.hstack((Demographic_graduated_doctorate_track, Graduate_doctorate_arr))

        # Total in university
        Total_univ_arr = np.zeros((2,1))
        for n_dem in range(2):
            idx_sum_total                       = np.logical_and(df_all_univ["MINRTY"]==n_dem, df_all_univ['INUNIV']==1)
            Total_univ_arr[n_dem]               = sum( idx_sum_total )
        Demographic_in_univ_track               = np.hstack((Demographic_in_univ_track, Total_univ_arr))

        # Total went undergrad
        Went_undergrad_arr = np.zeros((2,1))
        for n_dem in range(2):
            idx_went_undergrad                   = Went_Undergrad["MINRTY"]==n_dem
            Went_undergrad_arr[n_dem]            = sum( idx_went_undergrad )
        Demographic_went_undergrad_track         = np.hstack((Demographic_went_undergrad_track, Went_undergrad_arr))

        # Total went masters
        Went_masters_arr = np.zeros((2,1))
        for n_dem in range(2):
            idx_went_masters                    = Went_Masters["MINRTY"]==n_dem
            Went_masters_arr[n_dem]             = sum( idx_went_masters )
        Demographic_went_masters_track          = np.hstack((Demographic_went_masters_track, Went_masters_arr))

        # Total went doctorate
        Went_doctorate_arr = np.zeros((2,1))
        for n_dem in range(2):
            idx_went_doctorate                  = Went_Doctorate["MINRTY"]==n_dem
            Went_doctorate_arr[n_dem]           = sum( idx_went_doctorate )
        Demographic_went_doctorate_track        = np.hstack((Demographic_went_doctorate_track, Went_doctorate_arr))


        # University Updates
        df_all_univ["TIMEINUNIV"] += 1
        X_all_univ[:, feature_start_idxs['TIMEINUNIV'][1]]  = df_all_univ['TIMEINUNIV']/norm_params["const_div"][feature_start_idxs["TIMEINUNIV"][0]]

        # Workforce Updates
        df_emp["TIMEINWORKF"] += 1
        X_emp[:, feature_start_idxs['TIMEINWORKF'][1]] =  df_emp['TIMEINWORKF']/norm_params["const_div"][feature_start_idxs["TIMEINWORKF"][0]]

        # Update global variables
        X[idx_all_univ]             = X_all_univ
        df[idx_all_univ]            = df_all_univ
        X[idx_emp]                  = X_emp
        df[idx_emp]                 = df_emp
        
        idx_in_env = df["INENV"]==1
        df["AGE"][idx_in_env]               += 1
        df["TIMEINENV"][idx_in_env]         += 1
        N_students                          = sum(Demographic_in_univ_track[:,-1])
        df["N_STUDENTS_CURR"]               = N_students
        
        X[idx_in_env, feature_start_idxs['AGE'][1]]   = df['AGE'][idx_in_env]/norm_params["const_div"][feature_start_idxs["AGE"][0]]
        X[idx_in_env, feature_start_idxs['TIMEINENV'][1]]  = df['TIMEINENV'][idx_in_env]/norm_params["const_div"][feature_start_idxs["TIMEINENV"][0]]
        X[idx_in_env, feature_start_idxs['N_STUDENTS_CURR'][1]]  = df['N_STUDENTS_CURR'][idx_in_env]/norm_params["const_div"][feature_start_idxs["N_STUDENTS_CURR"][0]]

        #----------Update EnvTrackers----------
        EnvTrackers["Demographic_in_univ_track"]                = Demographic_in_univ_track
        EnvTrackers['Demographic_went_undergrad_track']         = Demographic_went_undergrad_track 
        EnvTrackers['Demographic_went_masters_track']           = Demographic_went_masters_track 
        EnvTrackers['Demographic_went_doctorate_track']         = Demographic_went_doctorate_track            
        EnvTrackers['Demographic_graduated_undergrad_track']    = Demographic_graduated_undergrad_track
        EnvTrackers['Demographic_graduated_masters_track']      = Demographic_graduated_masters_track
        EnvTrackers['Demographic_graduated_doctorate_track']    = Demographic_graduated_doctorate_track
        EnvTrackers["current_college_student"]                  = N_students
        
        return X, df, EnvTrackers


    def get_probability(self, df, degree):

        GPA     = df['UGGPA']
        TIME    = df['TIMEINUNIV']
        TUITION = df['ANNUALTUIT']

        if degree == 'undergrad':
            a1      = -1
            a21     = 0.5
            a22     = -0.05
            a3      = 0.0001
            bias    = 0
            
        elif degree == 'masters':
            a1      = -1
            a21     = 0.5
            a22     = -0.05
            a3      = 0.0001
            bias    = 1

        elif degree == 'doctorate':
            a1      = -1 
            a21     = 0.5
            a22     = -0.05
            a3      = 0.0001
            bias    = 0.5

        z = (a1*GPA + a21*TIME + a22*TIME**2 + a3*TUITION + bias).to_numpy().astype(float)
        prob = self.sig(z)

        return prob


    def UpdateNextGen(self, X, df, EnvTrackers):

        max_time = EnvTrackers["max_time_in_env"]
        norm_params = EnvTrackers["norm_params"]
        feature_start_idxs = EnvTrackers["feature_start_idxs"]

        # Replace people who have been in environment for max time in the population
        idx_done_in_env                     = df["TIMEINENV"] == max_time
        df["INENV"][idx_done_in_env]        = 0
        df["INUNIV"][idx_done_in_env]       = 0
        df["INWORKF"][idx_done_in_env]      = 0
        df["INMINTYPGRM"][idx_done_in_env]  = 0
        df["CURRENTGPA"][idx_done_in_env]   = 0
        df["ANNUALTUIT"][idx_done_in_env]   = 0
        df["TIMEINUNIV"][idx_done_in_env]   = 0
        df["TIMEINWORKF"][idx_done_in_env]  = 0
        df["TIMEINENV"][idx_done_in_env]    = 0
        df["DIVINVEST"][idx_done_in_env]    = 0
        df["AGE"][idx_done_in_env]          = 0
        df["SALARY"][idx_done_in_env]       = df[idx_done_in_env]["AVE_SALARY"]
        df["AVE_SALARY"][idx_done_in_env]   = 0

        X[idx_done_in_env, feature_start_idxs['INENV'][1]]       = 0
        X[idx_done_in_env, feature_start_idxs['INUNIV'][1]]      = 0
        X[idx_done_in_env, feature_start_idxs['INWORKF'][1]]     = 0
        X[idx_done_in_env, feature_start_idxs['INMINTYPGRM'][1]] = 0
        X[idx_done_in_env, feature_start_idxs['CURRENTGPA'][1]]  = 0
        X[idx_done_in_env, feature_start_idxs['ANNUALTUIT'][1]]  = 0
        X[idx_done_in_env, feature_start_idxs['TIMEINENV'][1]]   = 0
        X[idx_done_in_env, feature_start_idxs['TIMEINUNIV'][1]]  = 0
        X[idx_done_in_env, feature_start_idxs['TIMEINWORKF'][1]] = 0
        X[idx_done_in_env, feature_start_idxs['DIVINVEST'][1]]   = 0
        X[idx_done_in_env, feature_start_idxs['AGE'][1]]         = 0
        X[idx_done_in_env, feature_start_idxs['SALARY'][1]]      = X[idx_done_in_env, feature_start_idxs['AVE_SALARY'][1]]
        X[idx_done_in_env, feature_start_idxs['AVE_SALARY'][1]]  = 0

        # Sample people from the population
        X, df = self.sample_from_population(X, df, EnvTrackers)

        return X, df, EnvTrackers

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


    def UpdateRewardsTrackers(self, EnvTrackers):

        Demographic_applied_univ            = EnvTrackers["Demographic_applied_track"]
        Demographic_admitted_univ           = EnvTrackers["Demographic_admitted_track"]
        Demographic_went_to_undergrad       = EnvTrackers["Demographic_went_undergrad_track"]
        Demographic_graduated_undergrad     = EnvTrackers["Demographic_graduated_undergrad_track"]
        Demographic_went_to_masters         = EnvTrackers["Demographic_went_masters_track"]
        Demographic_graduated_masters       = EnvTrackers["Demographic_graduated_masters_track"]
        Demographic_went_to_doctorate       = EnvTrackers["Demographic_went_doctorate_track"]
        Demographic_graduated_doctorate     = EnvTrackers["Demographic_graduated_doctorate_track"]
        emp_profits                         = EnvTrackers["emp_profits_track"]
        Demographic_in_workforce            = EnvTrackers["Demographic_in_workforce_track"]
        Demographic_salary                  = EnvTrackers["Demographic_salary_track"]

        if Demographic_applied_univ.shape[1]>1:
            Demographic_applied_univ_curr           = Demographic_applied_univ[:,-1]
            Demographic_admitted_univ_curr          = Demographic_admitted_univ[:,-1]
            Demographic_went_to_undergrad_curr      = Demographic_went_to_undergrad[:,-1]
            Demographic_graduated_undergrad_curr    = Demographic_graduated_undergrad[:,-1]
            Demographic_went_to_masters_curr        = Demographic_went_to_masters[:,-1]
            Demographic_graduated_masters_curr      = Demographic_graduated_masters[:,-1]
            Demographic_went_to_doctorate_curr      = Demographic_went_to_doctorate[:,-1]
            Demographic_graduated_doctorate_curr    = Demographic_graduated_doctorate[:,-1]
            emp_profits_curr                        = emp_profits[-1]
            Demographic_in_workforce_curr           = Demographic_in_workforce[:,-1]
            Demographic_salary_curr                 = Demographic_salary[:,-1]

        else:

            Demographic_applied_univ_curr           = Demographic_applied_univ
            Demographic_admitted_univ_curr          = Demographic_admitted_univ
            Demographic_went_to_undergrad_curr      = Demographic_went_to_undergrad
            Demographic_graduated_undergrad_curr    = Demographic_graduated_undergrad
            Demographic_went_to_masters_curr        = Demographic_went_to_masters
            Demographic_graduated_masters_curr      = Demographic_graduated_masters
            Demographic_went_to_doctorate_curr      = Demographic_went_to_doctorate
            Demographic_graduated_doctorate_curr    = Demographic_graduated_doctorate
            emp_profits_curr                        = emp_profits
            Demographic_in_workforce_curr           = Demographic_in_workforce
            Demographic_salary_curr                 = Demographic_salary

        # Gather all rewards
        rewards = np.array([*Demographic_applied_univ_curr,
                            *Demographic_admitted_univ_curr,
                            *Demographic_went_to_undergrad_curr,
                            *Demographic_graduated_undergrad_curr,
                            *Demographic_went_to_masters_curr,
                            *Demographic_graduated_masters_curr,
                            *Demographic_went_to_doctorate_curr,
                            *Demographic_graduated_doctorate_curr,
                            emp_profits_curr,
                            *Demographic_in_workforce_curr,
                            *Demographic_salary_curr]).flatten()

        EnvTrackers["rewards"]  = rewards
    
        return EnvTrackers

#--------
    def State_Update(self, X_full, df_full, EnvTrackers, which_agent, time_step, k_cycle):

        # TODO: look for any specific features to remove/be filtered once this whole thing is up and running and we are honing in on details of the system

        # Incorporate and renormalize updates from df ---> X
        df_global = df_full["global"]
        X_global = X_full["global"]

        # University state/feature information
        if which_agent == "univ_adm":
            
            if time_step%k_cycle == 0:
                X_univ_adm, df_univ_adm = self.Global2AgentFeatures(X_global, df_global, EnvTrackers, "univ_adm", version="condensed")
            else:
                df_univ_adm  = None
                X_univ_adm   = None

            # Store insurance states in global storage
            df_full["univ_adm"]     = df_univ_adm
            X_full["univ_adm"]      = X_univ_adm

            return X_full, df_full

        if which_agent == "univ_budget":

            if time_step%k_cycle == 0:
                X_univ_budget, df_univ_budget = self.Global2AgentFeatures(X_global, df_global, EnvTrackers, "univ_budget", version="condensed")
            else:
                df_univ_budget  = None
                X_univ_budget   = None

            # Store in university states in global storage
            df_full["univ_budget"]      = df_univ_budget
            X_full["univ_budget"]       = X_univ_budget

            return X_full, df_full

        # Employer state/feature information
        if which_agent == "emp":
            if time_step%k_cycle == 0:
                X_emp, df_emp = self.Global2AgentFeatures(X_global, df_global, EnvTrackers, "emp", version="condensed")
                
            else:
                df_emp = None
                X_emp  = None

            # Store employer states in global storage
            df_full["emp"]                  = df_emp
            X_full["emp"]                   = X_emp

            return X_full, df_full

        # Omni. planner state/feature information
        if which_agent == "planner":
            if time_step%k_cycle == 0:
                df_planner              = df_global
                X_planner               = X_global
            else:
                df_planner  = None
                X_planner   = None

            # Store planner states in global storage
            df_full["planner"]      = df_planner
            X_full["planner"]       = X_planner

            return X_full, df_full
    

    
