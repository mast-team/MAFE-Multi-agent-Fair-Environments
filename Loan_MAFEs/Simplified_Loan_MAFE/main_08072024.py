import argparse
import torch
import os
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.buffer_MAPPO import ReplayBuffer as ReplayBuffer_MAPPO
from utils.buffer_MADDPG import ReplayBuffer as ReplayBuffer_MADDPG
from algorithms.multiagent_algos_082024 import CrossEntropyMethod, MAPPO, MADDPG
from make_env_04082024 import make_env_04082024
import glob
import _pickle as pickle
import gzip
import warnings


def get_agent_scenario(env_id, policyTypes):

    # Which agents are RL models (True) and which agents
    # part of the environment (False)

    which_agents = []

    if env_id == 'Loan_environment_fair_04082024':

        # Admission agent
        if policyTypes[0] == None:
            which_agents.append(False)
        else:
            which_agents.append(True)

        # Disbursement agent
        if policyTypes[1] == None:
            which_agents.append(False)
        else:
            which_agents.append(True)

        # Debt agent
        if policyTypes[2] == None:
            which_agents.append(False)
        else:
            which_agents.append(True)
            
    elif env_id == 'Healthcare_environment':

        tbd = "TO COME"

    return which_agents

def torch_2_numpy_actions(torch_agent_actions, empty_obs, model_type):

    ''' Convert actions to numpy arrays '''
    
    actions = []
    for i in range(len(torch_agent_actions)):
        if torch_agent_actions[i] is not None:
            
            if model_type == "MAPPO":
                action = torch_agent_actions[i][1].detach().numpy()
            else:
                action = torch_agent_actions[i].detach().numpy()

            # If empty observation, replace action with empty action
            if empty_obs[i] == True:
                NN = max(action.shape)
                action = np.zeros((0,NN))
                
            actions.append( np.clip(action, 0, 1) )
        else:
            actions.append( None )

    return actions

def preprocess_obs(obs, which_agents):

    empty_obs = np.array([False,False,False])
    torch_obs_fake = []
    torch_obs_true = []

    for i in range(len(obs)):
        if which_agents[i]:

            # Augment observation space with extra dimension (to account for empty observation states)
            # Extra dimension serves as a flag to let model know that we received no observation
            MM,NN = obs[i].shape
                
            if MM == 0:
                obs_4_train = np.zeros((1,NN+1))
                obs_aug = np.zeros((0,NN+1))
                obs_4_train[0,NN] = 1
                

                # Document empty obs for this agent
                empty_obs[i] = True
                    
            else:
                aug = np.zeros((MM,1))
                obs_aug = np.hstack((obs[i],aug))
                obs_4_train = obs_aug.copy()

            # Add torch obs
            torch_obs_fake.append(Variable(torch.Tensor(obs_4_train),requires_grad=False))
            torch_obs_true.append(Variable(torch.Tensor(obs_aug),requires_grad=False))
                
        else:
            torch_obs_fake.append(None)
            torch_obs_true.append(None)

    return torch_obs_true, torch_obs_fake, empty_obs


def compute_Gs_per_episode(comps, gamma, which_agents, lmds, eps=1e-10):

    # The rewards-to-go (rtg) per episode per batch to return
    Gs = []
 
    # Iterate through each episode backwards to maintain same order in batch_rtgs
    for i in range(len(comps)):
        batch_rtgs = []
        
        if which_agents[i]:

            discounted_reward = 0 # Discounted reward so far
            for comp in reversed(comps[i]):

                # Calculate rewards from component functions
                profits         = comp[0,0]
                applied_w       = comp[0,1]
                applied_nw      = comp[0,2]
                accepted_w      = comp[0,3]
                accepted_nw     = comp[0,4]
                wait_w          = comp[0,5]
                wait_nw         = comp[0,6]
                disbursed_w     = comp[0,7]
                disbursed_nw    = comp[0,8]
                default_w       = comp[0,9]
                default_nw      = comp[0,10]
                score_w         = comp[0,11]
                score_nw        = comp[0,12]
                terminated_w    = comp[0,13]
                terminated_nw   = comp[0,14]

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

                rew =  lmds[0]*profits_rewards + lmds[1]*adm_Rts_rewards + lmds[2]*default_Rts_rewards \
                          + lmds[3]*adm_Rts_Eqty_rewards + lmds[4]*wait_Rts_Eqty_rewards + lmds[5]*default_Rts_Eqty_Rts_rewards

                discounted_reward = rew + discounted_reward * gamma
                batch_rtgs.insert(0, discounted_reward)

        else:
            batch_rtgs = [None]*len(rews[i])

        Gs.append(batch_rtgs)

    return Gs

def get_log_probs(step_output, which_agents):
    
    log_probs = []
    for ii in range(len(step_output)):

        if which_agents[ii]:
            log_probs.append(step_output[ii][2])

        else:
            log_probs.append(None)

    return log_probs

def get_fake_actions(torch_agent_true_actions, empty_obs, model_type):

    ''' Fake actions only are required when the obs is empty. Otherwise,
        torch_agent_true_actions = torch_agent_fake_actions'''

    torch_agent_fake_actions = []

    for ii in range(len(empty_obs)):
        
        if torch_agent_true_actions[ii] is not None:

            if model_type == "MAPPO":
                MM,NN = torch_agent_true_actions[ii][1].shape
            else:
                MM,NN = torch_agent_true_actions[ii].shape
            
            if MM == 0:
                
                act_4_train = np.zeros((1,NN)) # Action of all zeros if no observations provided

            else:
                if model_type == "MAPPO":
                    act_4_train = torch_agent_true_actions[ii][1]
                else:
                    act_4_train = torch_agent_true_actions[ii]

            torch_agent_fake_actions.append(Variable(torch.Tensor(act_4_train),requires_grad=False))

        else:
            torch_agent_fake_actions.append(None)

    return torch_agent_fake_actions
    

# Code that handles model specific update goes here
def model_specific_update(env, model, obs, model_type, which_agents, other_params, replay_buffer=None):

    if model_type == 'CrossEntropyMethod':

        episode_length      = other_params['episode_length'] # How many time steps in episode
        n_episodes          = other_params['n_episodes']
        N_episodes          = other_params['N_episodes'] # How many episodes in an epoch
        epoch               = other_params['epoch']  # current epoch
        N_episodes          = other_params['N_episodes']
        N_epochs            = other_params['N_epochs']   # How many total training epochs
        tt                  = other_params['tt']
        store_rewards_dir   = other_params['store_rewards_dir']
        trial_num           = other_params['trial_num']

        # Start of an episode
        if tt == 0:

            print("Starting Episode %i of Epoch %i" % (n_episodes, epoch))
            
            # Reset World
            obs = env.reset_world()

            # Sample model parameters
            model.sample_intervention_parameters()

            # Update time step in episode
            tt += 1

            # Update parameters
            other_params['tt'] = tt
            return obs, env, model, other_params

        # Get actions
        actions = model.apply_intervention_model(obs)        

        # Import actions to environment
        if which_agents[0]:
            env.world.agents[0].action = actions[0]

            # Save actions
            if actions[0] is not None and len(actions[0].flatten()) > 0:
                model.ActionTrackers["adm_actions"][epoch, n_episodes, tt-1, :] = actions[0].flatten()
            else:
                N_sz = len(model.ActionTrackers["adm_actions"][epoch, n_episodes, tt-1, :])
                model.ActionTrackers["adm_actions"][epoch, n_episodes, tt-1, :] = np.ones(( N_sz ))*-1

        # If there is a disbursement agent, update its state
        if which_agents[1]:
            env.world.agents[1].action = actions[1]

        # If there is a debt agent, update its state
        if which_agents[2]:
            env.world.agents[2].action = actions[2]

            # Save actions
            if actions[2] is not None and len(actions[2].flatten()) > 0:
                model.ActionTrackers["debt_actions"][epoch, n_episodes, tt-1, :] = actions[2].flatten()
            else:
                N_sz = len(model.ActionTrackers["debt_actions"][epoch, n_episodes, tt-1, :])
                model.ActionTrackers["debt_actions"][epoch, n_episodes, tt-1, :] = np.ones(( N_sz ))*-1

        # Update environment
        obs, reward, full_reward, bank_default_flag,_ = env.step(tt-1)
        if tt>1:
            model.reward_t = np.vstack((model.reward_t, reward))
            model.full_reward_t = np.vstack((model.full_reward_t, full_reward))
        
        
        # Update time step in episode
        tt += 1

        # Check for bank failure (If failed, bad episode to training...restart)
        if bank_default_flag or (tt == episode_length and model.full_reward_t[-1,0]==0):
            model.reset_episode()
            tt = 0
            
        # End of an episode
        if (tt == episode_length and model.full_reward_t[-1,0]!=0):

            # Store rewards from this episode
            model.update_rewards()
            
            # Store sampled weights from current iteration for elite set calculation
            model.DictRealizations_N.append(model.AgentRealization)

            # Reset model for next episode
            model.reset_episode()
            n_episodes += 1
            
            tt = 0 # Envirnoment reset

        # End of epoch
        if n_episodes == N_episodes: 
            
            # Update Epoch
            model.step()
            model.reset_epoch()
            epoch += 1
            model.epoch = epoch
            n_episodes = 0

            # Model storage
            store_model_file = store_rewards_dir + "trial_"+ str(trial_num) + "/model.pkl"
            file_model = open(store_model_file, 'wb')
            pickle.dump(model, file_model)

        # Update parameters
        other_params['epoch']=epoch
        other_params['n_episodes'] = n_episodes
        other_params['tt'] = tt
        if epoch == N_epochs:
            other_params['termination_flag'] = True

        return obs, env, model, other_params

    if model_type == 'MADDPG':

        episode_length      = other_params['episode_length'] # How many time steps in episode
        n_episodes          = other_params['n_episodes']
        N_episodes          = other_params['N_episodes'] # How many episodes until training complete (make equal to N_episodes*epoch the value for CrossEntropyMethod)
        tt                  = other_params['tt']
        N_replay            = other_params['N_replay']
        N_replay_prev       = other_params['N_replay_prev']
        store_rewards_dir   = other_params['store_rewards_dir']
        trial_num           = other_params['trial_num']
        ep_len              = other_params['episode_length']
        buffer_length       = other_params['buffer_length']
        batch_size          = other_params['batch_size']
        steps_per_update    = other_params['steps_per_update']
        
        if model_type == 'MADDPG':
            explore = True
        else:
            explore = False

        # Start of an episode
        if tt == 0:

            print("Starting Episode %i" % (n_episodes))
            
            # Reset World
            obs = env.reset_world()

            # Update time step in episode
            tt += 1

            # Update parameters
            other_params['tt'] = tt

            return obs, env, model, other_params

        # Prep model inputs
        torch_obs_true, torch_obs_fake, empty_obs = preprocess_obs(obs.copy(), which_agents)

        # Get true actions as torch Variables
        torch_agent_true_actions = model.step(torch_obs_true, N_replay, explore=explore)  # ep_num is only updated for MADDPG not MAAC

        # Get "fake" actions (used for training when observations are empty for a given agent)
        torch_agent_fake_actions = get_fake_actions(torch_agent_true_actions, empty_obs, model_type)

        # Detach Actions
        torch_actions_detach_fake = []
        for aa in torch_agent_fake_actions:
            if aa is not None:
                torch_actions_detach_fake.append(aa.detach())
            else:
                torch_actions_detach_fake.append(None)

        # Convert torch to numpy arrays
        actions = torch_2_numpy_actions(torch_agent_true_actions, empty_obs, model_type)

        # -------------Import actions to environment-------------
        # If there is an admissions agent, update its state
        if which_agents[0]:
            env.world.agents[0].action = actions[0]

            # Save actions
            if actions[0] is not None and len(actions[0].flatten()) > 0:
                model.ActionTrackers["adm_actions"] = np.vstack((model.ActionTrackers["adm_actions"], actions[0].flatten()))
            else:
                model.ActionTrackers["adm_actions"] = np.vstack((model.ActionTrackers["adm_actions"], np.ones(( 2 ))*-1))

        # If there is a disbursement agent, update its state
        if which_agents[1]:
            env.world.agents[1].action = actions[1]

        # If there is a debt agent, update its state
        if which_agents[2]:
            env.world.agents[2].action = actions[2]

            # Save actions
            if actions[2] is not None and len(actions[2].flatten()) > 0:
                model.ActionTrackers["debt_actions"] = np.vstack((model.ActionTrackers["debt_actions"], actions[2].flatten()))
            else:
                model.ActionTrackers["debt_actions"] = np.vstack((model.ActionTrackers["debt_actions"], np.ones(( 2 ))*-1))

        #-------------------------------------------------------
        # Update environment and get next obs and reward
        next_obs, reward, full_reward, bank_default_flag, dones = env.step(tt-1)

        # Prep model inputs for next obs
        next_torch_obs_true, next_torch_obs_fake, next_empty_obs = preprocess_obs(next_obs.copy(), which_agents)

        model.reward_t = np.vstack((model.reward_t, reward))
        model.full_reward_t = np.vstack((model.full_reward_t, full_reward)) # Augmented to include more rewards. Later on, replace regular rewards with full rewards.

        # Add info to replay buffer
        if tt > 1:
            replay_buffer.push_candidates(torch_obs_fake, torch_actions_detach_fake, reward, next_torch_obs_true, next_torch_obs_fake, empty_obs, next_empty_obs, dones)

        # Update episode time step and obs for next time step
        obs = next_obs
        tt += 1

        if (N_replay >= int(buffer_length/25)) and (N_replay-N_replay_prev>=steps_per_update):             
            model.prep_training(device='cpu')
            sample = replay_buffer.sample(batch_size)

            # For plotting
            plot_path = './testing_in_environment/MADDPG/trial_'+str(trial_num)+'/'					
            if not os.path.exists(plot_path):                                                                       
                os.mkdir(plot_path)                                                                                
															
            Q_plot_file_name = plot_path +'Q_plots/'                                                                
            Action_plot_file_name = plot_path +'action_plots/'                                                    
            if os.path.exists(Q_plot_file_name):                                                                    
                plt_num = len(glob.glob1(Q_plot_file_name+'agent_0/',"plt_num_*"))                                                                          
            else:     
                os.mkdir(Q_plot_file_name)                                                                                              
                os.mkdir(Q_plot_file_name+'agent_0/') 
                os.mkdir(Q_plot_file_name+'agent_1/') 
                os.mkdir(Q_plot_file_name+'agent_2/')                                                                       
                plt_num  = 0                                              
               														
            if not os.path.exists(Action_plot_file_name):                                                                                    
                os.mkdir(Action_plot_file_name)                                                                                              
                os.mkdir(Action_plot_file_name + 'agent_0/') 
                os.mkdir(Action_plot_file_name + 'agent_1/') 
                os.mkdir(Action_plot_file_name + 'agent_2/')  
														
            model.update(sample, which_agents, batch_size, Q_plot_file_name, Action_plot_file_name, plt_num)       
            model.update_all_targets()
            model.prep_rollouts(device='cpu')
            N_replay_prev = N_replay

        # Check for episode end (Due to bank failure or terminal time step)
        if bank_default_flag or (tt == episode_length):

            #------------------Replay Buffer Adjustment------------------
            N_replay += len(replay_buffer.cand_obs_buffs[0])               
            replay_buffer.push()
            print("# Samples Trained On: "+str(N_replay))						
            print("Replay buffer length: "+str(len(replay_buffer)))					

            # Episode unsuccessful so remove the N_replay_curr samples from this episode that we added to the replay buffer
            replay_buffer.reset_candidates()
                
            # Rewards Storage
            model.update_rewards()

            # Model storage
            store_model_file = store_rewards_dir + "trial_"+ str(trial_num) + "/model.pkl"
            file_model = open(store_model_file, 'wb')
            pickle.dump(model, file_model)

            # Reset model for next episode
            model.reset_episode()

            # Update episode number
            n_episodes += 1
            model.n_episodes = n_episodes
            print("Num Episodes: "+str(n_episodes)) # print number of successful episodes

            tt = 0 

        # Update parameters
        other_params['n_episodes'] = n_episodes
        other_params['tt'] = tt
        other_params['N_replay'] = N_replay
        other_params['N_replay_prev'] = N_replay_prev
        if n_episodes == N_episodes:
            other_params['termination_flag'] = True

        return obs, env, model, other_params
    

    if model_type == 'MAPPO':
        
        episode_length      = other_params['episode_length'] # How many time steps in episode
        n_episodes          = other_params['n_episodes']
        N_episodes          = other_params['N_episodes'] # How many episodes until training complete (make equal to N_episodes*epoch the value for CrossEntropyMethod)
        N_replay            = other_params['N_replay']
        store_rewards_dir   = other_params['store_rewards_dir']
        trial_num           = other_params['trial_num']
        ep_len              = other_params['episode_length']
        batch_size          = other_params['batch_size']
        num_epochs          = other_params['num_epochs']
        N_ep_per_batch      = other_params['N_ep_per_batch']
        lmds                = other_params['lmds']
        lmds                = np.array(lmds).astype('float')

        # Initialize buffer for storing rollouts
        replay_buffer = ReplayBuffer_MAPPO()

        # Generate rollouts
        for ep_num in range(N_ep_per_batch):
            for tt in range(episode_length):

                # Start of an episode
                if tt == 0:

                    print("Starting Episode %i" % (ep_num))
            
                    # Reset World
                    obs = env.reset_world()

                    # Update time step in episode
                    continue

                # Prep model inputs
                torch_obs_true, torch_obs_fake, empty_obs = preprocess_obs(obs.copy(), which_agents)

                # Get true actions as torch Variables
                torch_agent_true_actions = model.step(torch_obs_true)  # ep_num is only updated for MADDPG not MAAC

                # Get log_probs
                log_probs = get_log_probs(torch_agent_true_actions, which_agents) 

                # Get "fake" actions (used for training when observations are empty for a given agent)
                torch_agent_fake_actions = get_fake_actions(torch_agent_true_actions, empty_obs, model_type)

                # Detach Actions
                torch_actions_detach_fake = []
                for aa in torch_agent_fake_actions:
                    if aa is not None:
                        torch_actions_detach_fake.append(aa.detach())
                    else:
                        torch_actions_detach_fake.append(None)

                # Convert torch to numpy arrays
                actions = torch_2_numpy_actions(torch_agent_true_actions, empty_obs, model_type)

                # -------------Import actions to environment-------------
                # If there is an admissions agent, update its state
                if which_agents[0]:
                    env.world.agents[0].action = actions[0]
                    print(actions[0])

                    # Save actions
                    if actions[0] is not None and len(actions[0].flatten()) > 0:
                        model.ActionTrackers["adm_actions"] = np.vstack((model.ActionTrackers["adm_actions"], actions[0].flatten()))
                    else:
                        model.ActionTrackers["adm_actions"] = np.vstack((model.ActionTrackers["adm_actions"], np.ones(( 2 ))*-1))

                # If there is a disbursement agent, update its state
                if which_agents[1]:
                    env.world.agents[1].action = actions[1]

                # If there is a debt agent, update its state
                if which_agents[2]:
                    env.world.agents[2].action = actions[2]
                    print(actions[2])

                    # Save actions
                    if actions[2] is not None and len(actions[2].flatten()) > 0:
                        model.ActionTrackers["debt_actions"] = np.vstack((model.ActionTrackers["debt_actions"], actions[2].flatten()))
                    else:
                        model.ActionTrackers["debt_actions"] = np.vstack((model.ActionTrackers["debt_actions"], np.ones(( 2 ))*-1))
                    
                #-------------------------------------------------------
                # Update environment and get reward
                next_obs, reward, full_reward, bank_default_flag, dones = env.step(tt)

                model.reward_t = np.vstack((model.reward_t, reward))

                # Add info to replay buffer
                if tt > 0: 				# First set of observations is just empty?*********
                    replay_buffer.push_candidates(torch_obs_fake, torch_actions_detach_fake, log_probs, reward, empty_obs, dones)

                # Update episode time step and obs for next time step
                obs = next_obs

                # Check for bank failure (If failed, unsuccessful training episode...restart)
                if bank_default_flag:
                    print(tt)
                    print(reward[0])
                    break
                               
            ### Compute Gs here and update buffer with them
            replay_buffer.cand_Gs_buffs = compute_Gs_per_episode(replay_buffer.cand_rew_buffs, model.gamma, which_agents, lmds)
    
            # End of Episode
            model.full_reward_t = np.vstack((model.full_reward_t, full_reward)) # Augmented to include more rewards. Later on, replace regular rewards with full rewards.
            
            # Rewards for non-failure episodes
            replay_buffer.push()                                           
            replay_buffer.reset_candidates()   									
            print("# Samples Trained On: "+str(N_replay))						
            print("Replay buffer length: "+str(len(replay_buffer)))					
   
        ####### Update model ######
        model.prep_training(device='cpu')
        for ii in range(num_epochs):

            sample_sz = min([batch_size, len(replay_buffer)])
            sample = replay_buffer.sample(sample_sz)
               
            plot_path = './testing_in_environment/MAPPO/trial_'+str(trial_num)+'/'					
            if not os.path.exists(plot_path):                                                                       
                os.mkdir(plot_path)                                                                                															
            Q_plot_file_name = plot_path +'Q_plots/'                                                                
            Action_plot_file_name = plot_path +'action_plots/'                                                      
            if os.path.exists(Q_plot_file_name):                                                                    
                plt_num = len(glob.glob1(Q_plot_file_name+'agent_0/',"plt_num_*"))                                                                          
            else:     
                os.mkdir(Q_plot_file_name)                                                                                              
                os.mkdir(Q_plot_file_name+'agent_0/') 
                os.mkdir(Q_plot_file_name+'agent_1/') 
                os.mkdir(Q_plot_file_name+'agent_2/')                                                                       
                plt_num  = 0                                              
               														
            if not os.path.exists(Action_plot_file_name):                                                                                    
                os.mkdir(Action_plot_file_name)                                                                                              
                os.mkdir(Action_plot_file_name + 'agent_0/') 
                os.mkdir(Action_plot_file_name + 'agent_1/') 
                os.mkdir(Action_plot_file_name + 'agent_2/')                                                                                                                      

            model.update(sample, which_agents, sample_sz , Q_plot_file_name, Action_plot_file_name, plt_num)     

                
        ###### Rewards Storage ###### (Last episode of the batch)
        model.update_rewards()
                
        ###### Model storage ######
        store_model_file = store_rewards_dir + "trial_"+ str(trial_num) + "/model.pkl"
        file_model = open(store_model_file, 'wb')
        pickle.dump(model, file_model)

        # Reset model for next episode
        model.reset_episode()

        # Update episode number
        n_episodes += 1
        print("Num Updates: "+str(n_episodes)) # print number of successful episodes
            
        # Update parameters
        other_params['n_episodes'] = n_episodes
        other_params['N_replay'] = N_replay
        if n_episodes == N_episodes:
            other_params['termination_flag'] = True

        return obs, env, model, other_params

def run(config):

    if config.SEED is not None:
        run_num = config.SEED
    else:
        run_num = 0

    which_agents = get_agent_scenario(config.env_id, config.policyTypes)
    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_env_04082024(config.env_id,
                   config.policyTypes,
                   which_agents,
                   config.AwareOrNot,
                   config.eq_flags,
                   config.adm_freq)
    
    env.reset_world()
    # agent obs and action space info
    # Will need to clearly think about this since this value is variable.
    save_path = []

    # Initialize parameter storage
    other_params = {}
        
    if config.model_type == 'CrossEntropyMethod':
        
        # Get trial number (nth time we tried training this model for experimental results)

        # If new trial create new directory for it
        store_rewards_dir = os.getcwd() + "/stored_rewards_data/Profits_fairness/CrossEntropyMethod/"

        if config.trial_num is None:
            
            trial_num = len(glob.glob1(store_rewards_dir,"trial_*"))
            os.mkdir(store_rewards_dir + "trial_"+ str(trial_num) + "/")

            track_path = store_rewards_dir + "trial_"+ str(trial_num) + "/" # <---- Incorporate saving functionality
            # Convert inputs to model specs
            model = CrossEntropyMethod(env              = env,
                                       policyTypes      = config.policyTypes,
                                       AwareOrNot       = config.AwareOrNot,
                                       lmds             = config.lmd_all,
                                       eq_flags         = config.eq_flags,
                                       track_path       = track_path,
                                       N_epochs         = config.N_epochs,
                                       episode_length   = config.episode_length,
                                       N_episodes       = config.N_episodes,
                                       opt_type         = config.opt_type)
            other_params['epoch'] = 0

            # Store meta data associated with this training session
            file_meta_dta = store_rewards_dir + "trial_"+ str(trial_num) + "/meta_dta.txt"
            with open(file_meta_dta,'w') as out:
                line0 = "Training Method: "+str(config.model_type)
                line1 = "Weights for each element of loss (e.g. lmds): "+str(config.lmd_all)
                line2 = "Episode length: "+str(config.episode_length)
                line3 = "Training method subtype: "+config.opt_type
                line4 = "Number of epochs: "+str(config.N_epochs)
                line5 = "Number of episodes per training update: "+str(config.N_episodes)
                line6 = "Policy Types: "+str(config.policyTypes)
                line7 = "Seed: " + str(run_num)
                out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(line0,line1,line2,line3,line4,line5,line6,line7))

        # If we want to import an old trial load the model    
        else:
            trial_num = config.trial_num
            file_name_model = store_rewards_dir + "trial_"+ str(trial_num) + "/model.pkl"
            file_model = open(file_name_model, 'rb')
            model = pickle.load(file_model)
            
            # Make sure agents selected by user match the ones saved by the model
            if config.policyTypes != model.policyTypes:
                ValueError("policyTypes incompatible with saved model")

            other_params['epoch'] = model.epoch
             
        # Reset model
        model.reset_epoch()
        model.reset_episode()

        # Store parameters
        other_params['termination_flag'] = False
        other_params['episode_length'] = config.episode_length
        other_params['N_episodes'] = config.N_episodes
        other_params['N_epochs'] = config.N_epochs
        other_params['n_episodes'] = 0
        other_params['tt'] = 0
        other_params['store_rewards_dir'] = store_rewards_dir
        other_params['trial_num'] = trial_num
        other_params['lmds'] = config.lmds
        obs = None


    elif config.model_type == "MADDPG":
        
        # Get trial number (nth time we tried training this model for experimental results)

        # If new trial create new directory for it
        store_rewards_dir = os.getcwd() +  "/stored_rewards_data/Profits_fairness/MADDPG/"
        if config.trial_num is None:
            
            trial_num = len(glob.glob1(store_rewards_dir,"trial_*"))
            os.mkdir(store_rewards_dir + "trial_"+ str(trial_num) + "/")

            track_path = store_rewards_dir + "trial_"+ str(trial_num) + "/" # <---- Incorporate saving functionality
            # Convert inputs to model specs
            model = MADDPG(env = env,
                           which_agents     = which_agents,
                           policyTypes      = config.policyTypes,
                           AwareOrNot       = config.AwareOrNot,
                           lmds             = config.lmds,
                           gamma            = config.gamma,
                           tau              = config.tau,
                           lr_critic        = config.lr_critic,
                           lr_policy        = config.lr_policy,
                           hidden_dim       = config.hidden_dim,
                           eq_flags         = config.eq_flags,
                           track_path       = track_path,
                           Fair_scheme      = config.Fair_scheme_AC,
                           buffer_length    = config.buffer_length)
 
            replay_buffer = ReplayBuffer_MADDPG(config.buffer_length)
            other_params['n_episodes'] = 0
            other_params['episode_length'] = np.array([])

            # Store meta data associated with this training session
            file_meta_dta = store_rewards_dir + "trial_"+ str(trial_num) + "/meta_dta.txt"
            with open(file_meta_dta,'w') as out:
                line0 = "Training Method: "+str(config.model_type)
                line1 = "Weights for each element of loss (e.g. lmds): "+str(config.lmd_all)
                line2 = "Episode length: "+str(config.episode_length)
                line3 = "Training method subtype: "+config.opt_type
                line4 = "Gamma: "+str(config.gamma)
                line5 = "Tau: "+str(config.tau)
                line6 = "lr_critic: "+str(config.lr_critic)
                line7 = "lr_policy: "+str(config.lr_policy)
                line8 = "buffer_length: "+str(config.buffer_length)
                line9 = "steps_per_update: "+str(config.steps_per_update)
                line10 = "batch_size: "+str(config.batch_size)
                line11 = "Policy Types: "+str(config.policyTypes)
                line12 = "Seed: " + str(run_num)
                out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(line0,line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12))
        
        # If we want to import an old trial load the model
        else:
            trial_num = config.trial_num
            file_name_model = store_rewards_dir + "trial_"+ str(trial_num) + "/model.pkl"

            file_model = open(file_name_model, 'rb')							
            model = pickle.load(file_model)								
            												
            # Make sure agents selected by user match the ones saved by the model			
            if config.policyTypes != model.policyTypes:						
                ValueError("policyTypes incompatible with saved model")			
                												       
            replay_buffer = ReplayBuffer_MADDPG(config.buffer_length)
            
            other_params['n_episodes'] = model.n_episodes
            
        # Reset model
        model.reset_episode()

        # Store parameters
        other_params['step_num'] = 0
        other_params['termination_flag'] = False
        other_params['episode_length'] = config.episode_length
        other_params['N_episodes'] = config.N_episodes
        other_params['buffer_length'] = config.buffer_length
        other_params['batch_size'] = config.batch_size
        other_params['tt'] = 0
        other_params['N_replay_prev'] = 0
        other_params['store_rewards_dir'] = store_rewards_dir
        other_params['trial_num'] = trial_num
        other_params['steps_per_update'] = config.steps_per_update
        other_params['lmds'] = config.lmds
        other_params['N_replay'] = 0
        obs = None

    elif config.model_type == "MAPPO":
        
        # Get trial number (nth time we tried training this model for experimental results)
        
        # If new trial create new directory for it
        store_rewards_dir = os.getcwd() +  "/stored_rewards_data/Profits_fairness/MAPPO/"
        if config.trial_num is None:

            trial_num = len(glob.glob1(store_rewards_dir,"trial_*"))
            os.mkdir(store_rewards_dir + "trial_"+ str(trial_num) + "/")

            track_path = store_rewards_dir + "trial_"+ str(trial_num) + "/" # <---- Incorporate saving functionality
            # Convert inputs to model specs
            model = MAPPO(env = env,
                           which_agents     = which_agents,
                           policyTypes      = config.policyTypes,
                           AwareOrNot       = config.AwareOrNot,
                           lmds             = config.lmds,
                           gamma            = config.gamma,
                           lr_critic        = config.lr_critic,
                           lr_policy        = config.lr_policy,
                           hidden_dim       = config.hidden_dim,
                           eq_flags         = config.eq_flags,
                           track_path       = track_path,
                           Fair_scheme      = config.Fair_scheme_AC)
 
            other_params['n_episodes'] = 0

            # Store meta data associated with this training session
            file_meta_dta = store_rewards_dir + "trial_"+ str(trial_num) + "/meta_dta.txt"
            with open(file_meta_dta,'w') as out:
                line0 = "Training Method: "+str(config.model_type)
                line1 = "Weights for each element of loss (e.g. lmds): "+str(config.lmd_all)
                line2 = "Episode length: "+str(config.episode_length)
                line3 = "Training method subtype: "+config.opt_type
                line4 = "Gamma: "+str(config.gamma)
                line5 = "lr_critic: "+str(config.lr_critic)
                line6 = "lr_policy: "+str(config.lr_policy)
                line7 = "Rollout buffer size: "+str(config.N_ep_per_batch)
                line8 = "Updates per rollout: "+str(config.num_epochs)
                line9= "batch_size: "+str(config.batch_size)
                line10= "Policy Types: "+str(config.policyTypes)
                line11= "Seed: " + str(run_num)
                line12= "Policy Types: "+str(config.policyTypes)
                out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(line0,line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12))
        
        # If we want to import an old trial load the model
        else:
            trial_num = config.trial_num
            file_name_model = store_rewards_dir + "trial_"+ str(trial_num) + "/model.pkl"                     

            file_model = open(file_name_model, 'rb')							
            model = pickle.load(file_model)								
            												
            # Make sure agents selected by user match the ones saved by the model			
            if config.policyTypes != model.policyTypes:						
                ValueError("policyTypes incompatible with saved model")			       
            
            other_params['n_episodes'] = 0
            
        # Reset model
        model.reset_episode()

        # Store parameters
        other_params['termination_flag'] = False
        other_params['episode_length'] = config.episode_length
        other_params['N_episodes'] = config.N_episodes
        other_params['batch_size'] = config.batch_size
        other_params['store_rewards_dir'] = store_rewards_dir
        other_params['trial_num'] = trial_num
        other_params['N_ep_per_batch'] = config.N_ep_per_batch
        other_params['num_epochs'] = config.num_epochs
        other_params['lmds'] = config.lmds
        other_params['N_replay'] = 0
        obs = None

    else:
        raise error

    ep_i = 0
    while not other_params['termination_flag']:

        if config.model_type == 'CrossEntropyMethod' or config.model_type == 'MAPPO':

            obs, env, model, other_params = model_specific_update(env, model, obs, config.model_type, which_agents, other_params)
            
        else:
            obs, env, model, other_params = model_specific_update(env, model, obs, config.model_type, which_agents, other_params, replay_buffer)        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("--model_type",
                        help="Name of directory to store " +
                             "model/training contents", default="CrossEntropyMethod", type=str)

    parser.add_argument("--episode_length", default=800, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--hidden_dim", default=8, type=int)
    parser.add_argument("--pol_hidden_dim", default=8, type=int)
    parser.add_argument("--critic_hidden_dim", default=8, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.01, type=float)
    parser.add_argument("--q_lr", default=0.01, type=float)
    parser.add_argument("--lr_critic", default=0.01, type=float)
    parser.add_argument("--lr_policy", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)

    # Needed arguments
    parser.add_argument("--policyTypes", default="1,1,2", type=str)
    parser.add_argument("--intervention_names", default="profits,adm_Rts,default_Rts,adm_Eqty_Rts,wait_Rts_Eqty,default_Rts_Eqty",
                            help="Names of the different interventions included in rewards")
    parser.add_argument("--lmds", default="0.16666,0.16666,0.16666,0.16666,0.16666,0.16666", type=str)
    parser.add_argument("--N_epochs", default=50, type=int)
    parser.add_argument("--N_episodes", default=100, type=int)
    parser.add_argument("--AwareOrNot", default=1, type=int)
    parser.add_argument("--eq_flags",default="0,0,0", type=str)
    parser.add_argument("--adm_freq", default=1, type=int)
    parser.add_argument("--trial_num", default="None", type=str)
    parser.add_argument("--opt_type", default="cooperate", type=str)

    parser.add_argument("--N_ep_per_batch", default=100, type=int,
                            help="Number of episodes in rollout buffer")

    # MAPPO specific parameters
    parser.add_argument("--num_epochs", default=24, type=int,
                            help="Number of episodes in rollout buffer")

    # MADDPG/MAAC specific parameters
    parser.add_argument("--buffer_length", default=int(125*800), type=int)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--steps_per_update", default=790, type=int)

    # Fairness stuff (to build into policy methods later)
    parser.add_argument("--Fair_scheme_AC", default=0, type=int,
                            help="0: Furong's method, 1: accumualted direct rates")
    parser.add_argument("--SEED", default="None", type=str)
    

    config = parser.parse_args()

    # Update list items
    config.policyTypes = config.policyTypes.split(',')
    config.intervention_names   = config.intervention_names.split(',')
    config.lmds = config.lmds.split(',')
    config.eq_flags = config.eq_flags.split(',')
    config.lmd_all              = {}

    # Interpret commandline inputs
    if config.trial_num == "None":
        config.trial_num = None
    else:
        config.trial_num = int(config.trial_num)

    if config.SEED == "None":
        config.SEED = None
    else:
        config.SEED = int(config.SEED)


    # Eq_flags
    for ii in range(len(config.eq_flags)):
        if config.eq_flags[ii] == '0':
            config.eq_flags[ii] = False
        else:
            config.eq_flags[ii] = True

    # Policy types
    for ii in range(len(config.policyTypes)):
        if config.policyTypes[ii] == 'None':
            config.policyTypes[ii] = None
        else:
            config.policyTypes[ii] = int(config.policyTypes[ii])

    # Gather lmds
    for ii in range(len(config.lmds)):
        if config.lmds[ii] == 'None':
            config.lmd_all[config.intervention_names[ii]] = None
        else:
            config.lmd_all[config.intervention_names[ii]] = float(config.lmds[ii])


    # Update boolean items
    config.AwareOrNot = config.AwareOrNot == 1

    warnings.filterwarnings('ignore')

    run(config)
