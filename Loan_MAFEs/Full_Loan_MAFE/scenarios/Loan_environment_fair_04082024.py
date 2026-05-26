import numpy as np
from multiagent.core_loan_fair_04082024 import World, AgentAdmit, AgentDisburse, AgentDebt, AllCustomerInformation, AllBankInformation, TrackingIndicators, EnvModelParameters
import os, sys
import pandas as pd
import pdb
from matplotlib import pyplot as plt
import seaborn as sns

sys.path.insert(0, '../Longterm/')

''' This script constructs the loan processing environment class used to sample observations from actions made by agents'''
class Environment(object):

    def __init__(self, policyTypes, which_agents, AwareOrNot, eq_flags, adm_freq, aug_state_flag=False):

        self.world = self.make_world(policyTypes, which_agents, AwareOrNot, eq_flags, adm_freq, aug_state_flag)
    
    def make_world(self, policyTypes, which_agents, AwareOrNot, eq_flags, adm_freq, aug_state_flag=False):

        world = World()
        # Set any world properties first

        world.AwareOrNot = AwareOrNot
        world.which_agents = which_agents
        world.eq_flags = eq_flags
        world.adm_freq = adm_freq

        world.agents = []
        
        world.agents.append(AgentAdmit(AwareOrNot))
        world.agents[0].policyType = policyTypes[0]
        if which_agents[0]:
            world.agents[0].agent_flag = True
        else:
            world.agents[0].agent_flag = False

        world.agents.append(AgentDisburse(AwareOrNot))
        world.agents[1].policyType = policyTypes[1]
        if which_agents[1]:
            world.agents[1].agent_flag = True            
        else:
            world.agents[1].agent_flag = False            

        world.agents.append(AgentDebt(AwareOrNot))
        world.agents[2].policyType = policyTypes[2]
        if which_agents[2]:
            world.agents[2].agent_flag = True
        else:
            world.agents[2].agent_flag = False

        world.allcustomerinformation = AllCustomerInformation()
        world.allbankinformation = AllBankInformation()
        world.trackingindicators = TrackingIndicators()
        world.envmodelparameters = EnvModelParameters()
        
        world.features_imported = ['loan_amnt', 'annual_inc', 'dti', 'mths_since_last_delinq', 'fico_range_low', 'fico_range_high', 'int_rate', 'installment', 'term']
        world.features_imported_sens = world.features_imported.copy()
        world.features_imported_sens.append("race") # white: 0, nw: 1
        world.nn = 200                        # Number of people applying each time step
        world.NOversample = 3000              # Size of population
        world.NPerPeriod = 50                 # Number of people whose funds disbursed each round
        world.perDefault = 0.1                # If customer falls behind by more than perDefault% on their total requested loan payments for more than 2 consecutive periods, they default.
        world.perDefaultRecovery = 0.9        # Percentage of remaining balance that an institution is able to make up from a default. For now, put p% of remaining balance. But we actually could learn this too from our data.
        world.trainLoadAdmission = 0          # 0: Load existing model for admission decision, 1: Train new model and then load it for admissions decision
        world.trainLoadPropensity = 0         # 0: Load existing model for propensity estimation, 1: Train new model and then load it for propensity estimation (Non-sensitive attribute version)
        world.trainLoadPropensitySens = 0     # 0: Load existing model for propensity estimation, 1: Train new model and then load it for propensity estimation (Sensitive attribute version)
        world.store_path = "env_dta/"
        world.save_dta_flag = False
        world.aug_state_flag = aug_state_flag

        # Import data and models that are part of the environment
        world = self.loadDtaModels(world)

        #self.reset_world(world)
        
        #world.store_path = "./stored_results_backup_checking_stability_individDebt_11212023/kernel3/" # Path to store results
        #world.loss_types = ['direct']            # Choose between: 1. 'direct': simplest method averages over all rewards terms 2. 'weight': downweights nearterm profits to allow for unfairness in the beginning and 3. 'alternate': alternated between profit and fairness rewards
        #world.converge_flag = True            # True: Prevent early convergence, False: Original algorithm
        #world.save_dta_flag = False
        #world.N_pretrain = [40, [], []]
        #world.AwareOrNot = True
        
        return world

    def loadDtaModels(self,world):


        if world.save_dta_flag: # If we want to train and save a new model
            X_train, y_train_adm, x_control_train, df_train, X_test, y_test_adm, x_control_test, df_test, feature_names, norm_params = load_lendclub_data() # For training loan approval model
            np.savetxt(world.store_path+'X_train.txt', X_train)
            np.savetxt(world.store_path+'y_train_adm.txt', y_train_adm)
            np.savetxt(world.store_path+'x_control_train.txt', x_control_train)
            df_train.to_pickle(world.store_path+"df_train.pkl")
            np.savetxt(world.store_path+'X_test.txt', X_test)
            np.savetxt(world.store_path+'y_test_adm.txt', y_test_adm)
            np.savetxt(world.store_path+'x_control_test.txt', x_control_test)
            df_test.to_pickle(world.store_path+"df_test.pkl")
            np.save(world.store_path+"feature_names.npy", feature_names)
            np.save(world.store_path+"norm_params.npy", norm_params)
        else:                   # If we want to just load an old model
            X_train = np.loadtxt(world.store_path+'X_train.txt')
            y_train_adm = np.loadtxt(world.store_path+'y_train_adm.txt')
            x_control_train = np.loadtxt(world.store_path+'x_control_train.txt')
            df_train = pd.read_pickle(world.store_path+"df_train.pkl")
            X_test = np.loadtxt(world.store_path+'X_test.txt')
            y_test_adm = np.loadtxt(world.store_path+'y_test_adm.txt')
            x_control_test = np.loadtxt(world.store_path+'x_control_test.txt')
            df_test = pd.read_pickle(world.store_path+"df_test.pkl")
            feature_names = np.load(world.store_path+"feature_names.npy").tolist()
            norm_params = np.load(world.store_path+"norm_params.npy", allow_pickle=True).item()

        # Random oversampling
        idx_oversample = np.random.randint(0,999,world.NOversample)
        X_train         = X_train[idx_oversample]
        y_train_adm     = y_train_adm[idx_oversample]
        x_control_train = x_control_train[idx_oversample]
        X_test          = X_test[idx_oversample]
        y_test_adm      = y_test_adm[idx_oversample]
        x_control_test  = x_control_test[idx_oversample]
        df_train        = df_train.iloc[idx_oversample]
        df_train.reset_index(inplace=True)
        df_test         = df_test.iloc[idx_oversample]
        df_test.reset_index(inplace=True)

        # Constraints on inputs/outputs of models
        world.envmodelparameters.eq_admit_flag = world.eq_flags[0]
        world.envmodelparameters.eq_disburse_flag = world.eq_flags[1]
        world.envmodelparameters.eq_debt_flag = world.eq_flags[2]
        world.envmodelparameters.norm_params = norm_params

        world.X_train = X_train
        world.y_train_adm = y_train_adm
        world.x_control_train = x_control_train
        world.df_train = df_train
        world.X_test = X_test
        world.y_test_adm = y_test_adm
        world.x_control_test = x_control_test
        world.df_test = df_test
        world.feature_names = feature_names
        world.norm_params = norm_params

        # Load or (Train+Load) model for admissions decisisons (Logistic regression)
        print("Loading and/or training admission model...")
        idx_adm = np.array([0,1,2,3,4,5,8,9,10])

        # Admissions data (sensitive aware)
        world.X_train_adm = world.X_train[:,idx_adm]     # 'loan_amnt', 'annual_inc', 'dti', 'mths_since_last_delinq', 'fico_range_low', 'fico_range_high', 'int_rate', 'installment', 'term'
        world.X_test_adm = world.X_test[:,idx_adm]

        world.norm_params['mu_train'] = np.delete(world.norm_params['mu_train'], [6,7])
        world.norm_params['mu_test'] = np.delete(world.norm_params['mu_test'], [6,7])
        world.norm_params['sig_train'] = np.delete(world.norm_params['sig_train'], [6,7])
        world.norm_params['sig_test'] = np.delete(world.norm_params['sig_test'], [6,7])

        # Admissions data (sensitive unaware)
        world.X_train_adm_sens = np.hstack((world.X_train_adm, np.expand_dims(world.x_control_train,1) ))
        world.X_test_adm_sens = np.hstack((world.X_test_adm, np.expand_dims(world.x_control_test,1) ))

        # Load in model weights for aware and unaware admissions data    
        if world.trainLoadAdmission == 0:
            training_data_adm = []
        else:
            training_data_adm = [world.X_train_adm, world.y_train_adm]
        W_adm_unsens,b_adm_unsens = self.EntranceDecisionModel(training_data_adm, 0) # parameters for logistic regression. Used for original acceptance decisions from pool (unaware of sensitive attribute)
        W_adm_sens,b_adm_sens = self.EntranceDecisionModel(training_data_adm, 1) # parameters for logistic regression. Used for original acceptance decisions from pool (aware of sensitive attribute)
        world.envmodelparameters.W_adm_unsens = W_adm_unsens
        world.envmodelparameters.b_adm_unsens = b_adm_unsens
        world.envmodelparameters.W_adm_sens = W_adm_sens
        world.envmodelparameters.b_adm_sens = b_adm_sens
        
        if world.AwareOrNot:
            world.envmodelparameters.W_adm = W_adm_sens
            world.envmodelparameters.b_adm = b_adm_sens
        else:
            world.envmodelparameters.W_adm = W_adm_unsens
            world.envmodelparameters.b_adm = b_adm_unsens

        # Type of queue we use for monitoring people waiting for funds
        world.envmodelparameters.QueueType = 'Fair'

        # Plot score distributions
        #s_adm_orig = self.logistic(W_adm_sens,b_adm_sens,world.X_test_adm_sens)
        #self.plot_score_dists(s_adm_orig, x_control_test)

        # Propensity data + model (aware) We let the ground truth outcome for a person always be the aware more + noise. But the bank may test out aware vs unaware model
        print("Loading and/or training propensity score model...")
        world.feature_names_prop_sens = world.features_imported_sens
        world.idx_prop_install_sens = (np.array(world.feature_names_prop_sens)=='installment')
        X_train_prop_sens, y_train_prop_sens, X_test_prop_sens, y_test_prop_sens = self.LoadPropensitydata(world.df_train, world.df_test, world.feature_names_prop_sens)
        world.X_train_prop_sens = X_train_prop_sens
        world.y_train_prop_sens = y_train_prop_sens
        world.X_test_prop_sens = X_test_prop_sens
        world.y_test_prop_sens = y_test_prop_sens

        if world.trainLoadPropensitySens == 0:
            training_data_prop_sens = []
        else:
            training_data_prop_sens = [world.X_train_prop_sens, world.y_train_prop_sens]
        W_prop_sens,b_prop_sens = self.PropensityModel_fixed(training_data_prop_sens, 1) # parameters for logistic regression. Used for original acceptance decisions from pool
        world.envmodelparameters.W_prop_sens = W_prop_sens
        world.envmodelparameters.b_prop_sens = b_prop_sens

        # Propensity data + model (unaware)
        print("Loading and/or training propensity score model (WITH SENSITIVE ATTRIBUTE)...")
        world.feature_names_prop = world.features_imported
        world.idx_prop_install = (np.array(world.feature_names_prop)=='installment')
        X_train_prop, y_train_prop, X_test_prop, y_test_prop = self.LoadPropensitydata(world.df_train, world.df_test, world.feature_names_prop)

        world.X_train_prop = X_train_prop
        world.y_train_prop = y_train_prop
        world.X_test_prop = X_test_prop
        world.y_test_prop = y_test_prop

        if world.trainLoadPropensity == 0:
            training_data_prop = []
        else:
            training_data_prop = [X_train_prop,y_train_prop]
        W_prop,b_prop = self.PropensityModel_fixed(training_data_prop, world.AwareOrNot) # parameters for logistic regression. Used for original acceptance decisions from pool

        world.envmodelparameters.W_prop = W_prop
        world.envmodelparameters.b_prop = b_prop

        return world

    # Do this before each episode starts
    def reset_world(self):

        # Set the population to original features
        self.world.allcustomerinformation.df_pop = self.world.df_test
        self.world.allcustomerinformation.X_pop = {"adm":self.world.X_test_adm_sens.copy(),   "prop":self.world.X_test_prop_sens.copy()}

        # For sampling population applicants
        N_pop, k_feat = self.world.allcustomerinformation.X_pop['adm'].shape
        self.world.allcustomerinformation.linPop = [x for x in range(N_pop)]

        if self.world.AwareOrNot: # Aware
            self.world.allcustomerinformation.pop_scores = self.logistic(self.world.envmodelparameters.W_adm, self.world.envmodelparameters.b_adm, self.world.allcustomerinformation.X_pop["adm"])
            self.world.allcustomerinformation.scores = np.zeros( (0, self.world.allcustomerinformation.pop_scores.shape[1]) )
        else: # Unaware
            self.world.allcustomerinformation.pop_scores  = self.logistic(self.world.envmodelparameters.W_adm, self.world.envmodelparameters.b_adm, self.world.allcustomerinformation.X_pop["adm"][:,:-1])
            self.world.allcustomerinformation.scores = np.zeros( (0, self.world.allcustomerinformation.pop_scores.shape[1]) )

        #-------Sample of applicants-------
        self.world.allcustomerinformation.df_sample = pd.DataFrame(data=None, columns=self.world.allcustomerinformation.df_pop.columns)
        self.world.allcustomerinformation.X_sample_adm = np.zeros((0,k_feat))
        self.world.allcustomerinformation.X_sample_prop = np.zeros((0,k_feat))
        self.world.allcustomerinformation.df_sample_curr = pd.DataFrame(data=None, columns=self.world.allcustomerinformation.df_pop.columns)
        self.world.allcustomerinformation.X_sample_adm_curr = np.zeros((0,k_feat))
        self.world.allcustomerinformation.X_sample_prop_curr = np.zeros((0,k_feat))
        self.world.allcustomerinformation.PaymentPool = np.zeros((0,len(self.world.features_imported)+5+1))   # Information on existing loan (consists of 7 features: 1. customerID, 2. remaining balance on loan, 3. interest rate on loan, 4. time remaining to pay off loan, 5. Total payment requested, 6. Total amount paid, 7. Flag indicating whether account is in danger of default

        # People in loan pool
        self.world.allcustomerinformation.sampleIDs = np.array([])
        self.world.allcustomerinformation.df_pool = pd.DataFrame(data=None, columns=self.world.allcustomerinformation.df_pop.columns)
        self.world.allcustomerinformation.X_pool_adm  = np.zeros((0,k_feat))
        self.world.allcustomerinformation.X_pool_prop  = np.zeros((0,k_feat))
        self.world.allcustomerinformation.X_pool = {"adm":self.world.allcustomerinformation.X_pool_adm, "prop":self.world.allcustomerinformation.X_pool_prop}
        self.world.envmodelparameters.Q = np.zeros((5,0))                          # Queue for storing ordering of loan payouts
        self.world.allbankinformation.bank_outstanding_loans = np.zeros((0,3))     # Bank debts (deposits used to pay for loans + interest paid on those deposits. We abstract out the form of the deposits and just assume they all apply the same interest rate)
        self.world.allbankinformation.bank_total_debt = 0
        self.world.allbankinformation.bank_next_payment = 0
        #self.world.allcustomerinformation.Potential_earnings_arr  = np.zeros((0,1))

        ds_t = self.calcPopScoreDisparity(self.world.envmodelparameters.W_adm_sens, self.world.envmodelparameters.b_adm_sens, self.world.allcustomerinformation.X_pop["adm"], self.world.allcustomerinformation.df_pop)
        self.world.trackingindicators.reward = np.zeros((0,17))                                   # will append reward from each round and average over time steps
        self.world.trackingindicators.full_reward = np.zeros((0,5))
        self.world.trackingindicators.Potential_earnings = [0]
        self.world.trackingindicators.totals_4_installments = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]) 

        # Total profits that could have been been potentially earned from all people who applied for loans
        self.world.trackingindicators.Potential_earnings_arr = np.array([[0, -1, -2]])

        # System indicators to monitor every round
        self.world.trackingindicators.N_Applied = 0;                    # Total loans applicants system has received
        self.world.trackingindicators.N_Applied_w = 0;
        self.world.trackingindicators.N_Applied_nw = 0;
        self.world.trackingindicators.N_Applied_static = 0;           # Number of applications in current time step
        self.world.trackingindicators.N_Applied_w_static= 0;
        self.world.trackingindicators.N_Applied_nw_static = 0;

        self.world.trackingindicators.N_Score_Sum = 0               # Sum of scores of all applicants that ever aplied to system
        self.world.trackingindicators.N_Score_Sum_w = 0
        self.world.trackingindicators.N_Score_Sum_nw = 0
        self.world.trackingindicators.N_Score_Sum_static = 0        # Sum of scores of all applicants that aplied to system in current time step
        self.world.trackingindicators.N_Score_Sum_w_static = 0
        self.world.trackingindicators.N_Score_Sum_nw_static = 0

        self.world.trackingindicators.N_Accepted = 0;                 # Total applicants accepted into system
        self.world.trackingindicators.N_Accepted_w = 0
        self.world.trackingindicators.N_Accepted_nw = 0
        self.world.trackingindicators.N_Accepted_static = 0;          # Number of applicants accepted into system at current time step
        self.world.trackingindicators.N_Accepted_w_static = 0
        self.world.trackingindicators.N_Accepted_nw_static = 0
        
        self.world.trackingindicators.N_Disbursed = 0;                # Total loans disbursed to customers
        self.world.trackingindicators.N_Disbursed_w = 0;                
        self.world.trackingindicators.N_Disbursed_nw = 0;
        self.world.trackingindicators.N_Disbursed_static = 0;         # Number of customers with funds disbursed at current time step
        self.world.trackingindicators.N_Disbursed_w_static = 0;                
        self.world.trackingindicators.N_Disbursed_nw_static = 0;
        
        self.world.trackingindicators.N_Defaults = 0;                 # Total loans defaulted on by customers
        self.world.trackingindicators.N_Defaults_w = 0;
        self.world.trackingindicators.N_Defaults_nw = 0;
        self.world.trackingindicators.N_Defaults_static = 0;          # Number loans terminated in default in current time step
        self.world.trackingindicators.N_Defaults_w_static = 0;
        self.world.trackingindicators.N_Defaults_nw_static = 0;
        
        self.world.trackingindicators.N_Repaid = 0;                   # Total loans repaid by customers
        self.world.trackingindicators.N_Repaid_w = 0;
        self.world.trackingindicators.N_Repaid_nw = 0;
        self.world.trackingindicators.N_Repaid_static = 0;            # Number of loans terminated in repayment in current time step
        self.world.trackingindicators.N_Repaid_w_static = 0;
        self.world.trackingindicators.N_Repaid_nw_static = 0;

        self.world.trackingindicators.N_Terminated_w = 0;             # Total number of terminated loans
        self.world.trackingindicators.N_Terminated_nw = 0;

        self.world.trackingindicators.WaitTimes_w = 0                 # Total wait time of all people to receive their loans   
        self.world.trackingindicators.WaitTimes_nw = 0
        self.world.trackingindicators.WaitTimes_w_static = 0
        self.world.trackingindicators.WaitTimes_nw_static = 0
        
        self.world.trackingindicators.Profits = 0;
        self.world.trackingindicators.Profits_prev = 0
        self.world.allbankinformation.bank_default_flag = False

        #self.world.allbankinformation.bank_reject_all_flag = False
        self.world.trackingindicators.WaitTimes = {"disbursed_w":[],"disbursed_nw":[],"non_disbursed_w":[],"non_disbursed_nw":[]};

        # Set up initial state spaces
        obs = self.state_init(self.world)
        reward = self.world.trackingindicators.reward
        full_reward = self.world.trackingindicators.full_reward

        return obs

    def plot_score_dists(self, s, x_control, save=False):

        # Display each race's histograms
        s_nw = s[x_control==0]
        s_w = s[x_control==1]

        fig, axes = plt.subplots(1, 1)
        sns.distplot(s_nw, ax=axes, hist=False, kde = True, kde_kws = {'shade':True, 'linewidth': 3}, label='Non-White')
        sns.distplot(s_w, ax=axes, hist=False, kde = True, kde_kws = {'shade':True, 'linewidth': 3, 'linestyle':'--'}, label='White')
    
        axes.legend(prop={'size': 16})
        axes.set_xlabel('Score')
        axes.set_ylabel('Distribution')
        axes.set_title('Score Distributions')
        axes.set_xlim(-0.1,1.1)
        axes.set_ylim(0,5)
        plt.tight_layout()
    
        if save==False:
            plt.show()
            fig = 0

        return fig

    def step(self, iter_num):

        return self.world.step(iter_num)
    
    def state_init(self, world):

        _, N1 = world.allcustomerinformation.X_pool_adm.shape
        _, N2 = world.allcustomerinformation.PaymentPool.shape
        N_noSens = 1-int(world.AwareOrNot)# Remove sensitive attribute dimension
        obs = []
        
        # Initialize each state
        world.agents[0].state = np.zeros((0, N1 + 6 - N_noSens))
        world.agents[1].state = np.zeros((0, N1 + 7 - N_noSens))
        world.agents[2].state = np.zeros((0, N2 + 4 - N_noSens))

        # No state augmentation
        if world.aug_state_flag==False:
            obs.append(world.agents[0].state)
            obs.append(world.agents[1].state)
            obs.append(world.agents[2].state)

        # Augmented states
        else:
            _, N_s0 = world.agents[0].state.shape
            _, N_s1 = world.agents[1].state.shape
            _, N_s2 = world.agents[2].state.shape

            world.agents[0].aug_state = np.zeros((0, N_s0 + N_s1 + N_s2 + 2))
            world.agents[1].aug_state = np.zeros((0, N_s1 + N_s0 + N_s2 + 2))
            world.agents[2].aug_state = np.zeros((0, N_s2 + N_s0 + N_s1 + 2))
            
            obs.append(world.agents[0].aug_state)
            obs.append(world.agents[1].aug_state)
            obs.append(world.agents[2].aug_state)
        
        return obs


    def EntranceDecisionModel(self, training_data, AwareOrNot): 

        if training_data == []: # Train acc is 0.7885

            if AwareOrNot == False:
                ww = np.array([[ 0.32926726], [ 0.32099345], [-0.07767988], [ 0.05760095], [ 0.17960396], [ 0.01214556], [-0.48492992], [-0.35012773], [-0.42509067]]) 
                bb = np.array([1.1185488])
            else:
                ww = np.array([[0.45042902],[0.22925788],[-0.1226823],[0.0351262],[0.10676884],[0.08993863],[-0.440407],[-0.47313488],[-0.5100488],[-1.1442106]])
                bb = np.array([1.5128411])
            
            return ww, bb

        X_train, y_train = training_data
        X_train = X_train.astype(np.float32)
        y_train = np.expand_dims(y_train,1).astype(np.float32)
        y_train[y_train==-1]=0

        # Placeholders for training
        X = tf.placeholder(tf.float32, [None, X_train.shape[1]])
        y = tf.placeholder(tf.float32, [None, 1])

        # Define model
        with tf.variable_scope('LOG'):
            W = tf.get_variable('W', [X_train.shape[1],1],initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('b',[1],initializer=tf.constant_initializer(0.0))
            X_score = tf.matmul(X, W)
            X_prob = tf.sigmoid(X_score+b)

        X_pred = tf.where(X_prob>0.5, x=tf.ones_like(X_prob,tf.int32),y=tf.zeros_like(X_prob,tf.int32))
        loss=-tf.reduce_mean(y*tf.log(X_prob+1e-7)+(1-y)*tf.log(1-X_prob+1e-7))
    
        # Training performance
        correct_pred = tf.equal(X_pred, tf.cast(y,tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

        # Training parameters
        global_step = tf.Variable(0, trainable=False)
        boundaries = [500,1000,1500, 2500, 6500, 100000]
        values = [0.9, 0.1, 0.01,0.001,0.0001, 0.001, 0.0001]
        lr = tf.train.piecewise_constant(global_step, boundaries, values)
        train_opt = tf.train.MomentumOptimizer(lr,0.9)
        grads = train_opt.compute_gradients(loss)
        train_step = train_opt.apply_gradients(grads,global_step)

        sess=tf.Session()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        train_dict={X: X_train, y: y_train}

    
        for i in range(10000):
            _,train_prob,train_acc,cost,ww,bb = sess.run([train_step,X_prob,accuracy,loss,W,b],feed_dict=train_dict)
            if (i+1)%50==0:
                print("Epoch %d, loss %g, acc %g" % (i+1, cost, train_acc))

        return ww,bb

    
    def LoadPropensitydata(self, df_train, df_test, feature_names_prop):
    
        # Load or (Train+Load) model for predicting proensity of customer
        #feature_names_prop = ['loan_amnt', 'annual_inc', 'dti', 'mths_since_last_delinq', 'int_rate', 'term', 'installment']
        X_train_prop = df_train[feature_names_prop].to_numpy()
        X_test_prop = df_test[feature_names_prop].to_numpy()

        # Convert annual to monthly interest rates (amortize them)
        X_train_prop[:,6] = X_train_prop[:,6]/12
        X_test_prop[:,6] = X_test_prop[:,6]/12

        y_train_prop = ((df_train["total_rec_prncp"]+df_train["total_rec_int"])/df_train["total_pymnt"]).values
        y_test_prop = ((df_test["total_rec_prncp"]+df_test["total_rec_int"])/df_test["total_pymnt"]).values

        y_train_prop = np.clip(y_train_prop,0,1)
        y_test_prop = np.clip(y_test_prop,0,1)

        return X_train_prop, y_train_prop, X_test_prop, y_test_prop


    def PropensityModel_fixed(self, training_data, AwareOrNot):

        if training_data == []:
            if AwareOrNot == False:
                ww = np.array([ [ 3.49424134e-06], [ 6.87758836e-08], [-1.90435178e-06], [ 4.80396278e-04], [-4.77766414e-03], [ 4.80800033e-03], [-7.47039255e+00], [-1.06165709e-04], [-2.57993058e-03] ])
                bb = np.array([1.06944717])
            else:
                ww = np.array([[ 3.49731281e-06], [ 2.15853641e-08], [-1.67490507e-04], [ 3.02283671e-04], [-2.38987725e-02], [ 2.38959957e-02], [-6.32918476e+00], [-1.08707673e-04], [-2.56660198e-03], [-5.67529989e-02]])
                bb = np.array([1.04110082])
            
            return ww, bb

        X_train, y_train = training_data
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
    
        ww = np.array([[regr.coef_[0]],[regr.coef_[1]],[regr.coef_[2]],[regr.coef_[3]],[regr.coef_[4]],[regr.coef_[5]], [regr.coef_[6]], [regr.coef_[7]], [regr.coef_[8]], [regr.coef_[9]]])
        bb = np.array([regr.intercept_])

        return ww,bb

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